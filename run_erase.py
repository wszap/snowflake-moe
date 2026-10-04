# -*- coding: utf-8 -*-
"""任务3 可擦除性（版本 20001）

流程：
3.1 加载 v3_base_298k.pt，测莎士比亚基线 PPL（≈12.78）
3.2 抹除 50% 忆点（每细胞随机 50% 忆点 key+assembly 置零）→ 测 PPL → 存 v3_erased_50.pt
3.3 抹除 100% 忆点（全部置零）→ 测 PPL → 存 v3_erased_100.pt
3.4 重训恢复：冻结未擦除忆点，重训擦除忆点 + 其余参数（细胞器/编码器/路由等）5ep，测 PPL
3.5 输出 output/results_snowflake_erase.csv
红线：不改架构、不调 lambda、GPU<80°C（超则暂停 10 分钟）、每阶段跑完 empty_cache+gc
"""
import csv
import gc
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from snowflake_moe import HierarchicalCellMoE_LM  # noqa: E402
from train_lm import load_shakespeare, lm_batch, eval_ppl  # noqa: E402

CKPT_PATH = os.path.join(BASE, "checkpoints", "v3_base_298k.pt")
ERASE50_CKPT = os.path.join(BASE, "checkpoints", "v3_erased_50.pt")
ERASE100_CKPT = os.path.join(BASE, "checkpoints", "v3_erased_100.pt")
OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output",
                                       "results_snowflake_erase.csv"))

SEQ_LEN, BATCH_SIZE, EPOCHS, LR, SEED = 64, 128, 5, 3e-4, 2026

LAMBDA_ENT = 0.05
LAMBDA_MEM = 0.05
LAMBDA_ORG = 0.05


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def check_gpu_temp(max_c=79, wait_s=600):
    if DEVICE != "cuda":
        return
    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"], text=True)
            temp = int(out.strip().splitlines()[0])
            if temp < max_c:
                return
            print(f"[TEMP] GPU {temp}°C >= {max_c}°C，暂停 {wait_s}s")
            time.sleep(wait_s)
        except Exception as e:  # noqa: BLE001
            print(f"[TEMP] 查询失败，继续: {e}")
            return


def erase_memories(model, ratio, seed=SEED):
    """按比例抹除忆点（key+assembly 置零）。返回 (masked_count, total)。"""
    g = torch.Generator().manual_seed(seed + int(ratio * 100))
    total, masked = 0, 0
    for layer in model.layers:
        for cell in layer.cells:
            n = cell.memory_keys.shape[0]
            k = int(round(n * ratio))
            idx = torch.randperm(n, generator=g)[:k]
            with torch.no_grad():
                cell.memory_keys[idx] = 0.0
                cell.memory_assembly[idx] = 0.0
            total += n
            masked += k
    print(f"[erase] 抹除忆点: {masked}/{total} ({ratio:.0%})")
    return masked, total


def freeze_unerased(model, erased_idx_by_cell):
    """重训阶段：冻结未擦除忆点，其余参数（含擦除忆点）可训。"""
    for p in model.parameters():
        p.requires_grad_(True)
    for li, layer in enumerate(model.layers):
        for ci, cell in enumerate(layer.cells):
            keep = erased_idx_by_cell[li][ci]        # 擦除忆点下标
            n = cell.memory_keys.shape[0]
            free = torch.zeros(n, dtype=torch.bool)
            free[keep] = True
            for i in range(n):
                if not free[i].item():
                    cell.memory_keys.data[i] = cell.memory_keys.data[i].detach()
            # 直接按 Parameter 整体控制不可行（同一 Parameter 内要细粒度），
            # 改为：未擦除忆点记下原值并在每步前强制还原 + 置 requires_grad=False 的
            # 等价实现：这里采用"把未擦除 key/asm 移出可训集合"的最小代价方案——
            # 即：保存原值，每步前覆盖回去，且训练时不更新（手工还原法）。
            # 为清晰，下面用 requires_grad 掩码不可行，改为手动覆盖方案：
            pass
    return


def build_erase_index(model, ratio, seed=SEED):
    """返回每 cell 被抹除忆点的下标列表（与 erase_memories 同 seed 同逻辑）。"""
    g = torch.Generator().manual_seed(seed + int(ratio * 100))
    idx_map = []
    for layer in model.layers:
        layer_idx = []
        for cell in layer.cells:
            n = cell.memory_keys.shape[0]
            k = int(round(n * ratio))
            idx = torch.randperm(n, generator=g)[:k]
            layer_idx.append(idx)
        idx_map.append(layer_idx)
    return idx_map


class FrozenMemoryGuard:
    """冻结未擦除忆点：每步训练前把未擦除忆点覆盖回原值（等价于不参与梯度更新）。"""

    def __init__(self, model, keep_values):
        self.model = model
        self.keep = keep_values        # [(li, ci, tensor_idx, orig_key, orig_asm)]

    def restore(self):
        with torch.no_grad():
            for li, ci, idx, ok, oa in self.keep:
                cell = self.model.layers[li].cells[ci]
                cell.memory_keys.data[idx] = ok
                cell.memory_assembly.data[idx] = oa


def collect_keep_values(model, erased_idx_by_cell):
    keep = []
    for li, layer in enumerate(model.layers):
        for ci, cell in enumerate(layer.cells):
            erased = set(erased_idx_by_cell[li][ci].tolist())
            n = cell.memory_keys.shape[0]
            idx = torch.tensor([i for i in range(n) if i not in erased],
                               dtype=torch.long, device=cell.memory_keys.device)
            if idx.numel() == 0:
                continue
            keep.append((li, ci, idx,
                         cell.memory_keys.data[idx].clone(),
                         cell.memory_assembly.data[idx].clone()))
    return keep


def train_recover(model, train_ids, vocab_size, guard, tag):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_steps = max(1, train_ids.numel() // (SEQ_LEN * BATCH_SIZE))
    total_steps = n_steps * EPOCHS
    warmup = max(1, int(total_steps * 0.05))

    def lr_at(step):
        return LR * (step + 1) / warmup if step < warmup else LR

    t0 = time.time()
    step_global = 0
    for epoch in range(1, EPOCHS + 1):
        check_gpu_temp()
        model.train()
        e_sum, e_n = 0.0, 0
        for _ in range(n_steps):
            guard.restore()
            xb, yb = lm_batch(train_ids, BATCH_SIZE, SEQ_LEN, seed=SEED + step_global)
            for g in opt.param_groups:
                g['lr'] = lr_at(step_global)
            opt.zero_grad(set_to_none=True)
            logits, info = model(xb)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            loss = ce_loss
            # 沿用任务1的机制正则（HierarchicalCellMoE 多层 cell_infos 聚合）
            w = torch.cat([ci['weights'] for ci in info['cell_infos']], dim=0)
            attn = torch.cat([ci['memory_attn'] for ci in info['cell_infos']], dim=0)
            entropy = -(w * torch.log(w + 1e-9)).sum(-1).mean()
            memory_usage = attn.mean(dim=0)
            aux_memory = (memory_usage * memory_usage).sum() * attn.shape[-1]
            freq_soft = w.mean(dim=0)
            aux_organelle = (freq_soft * freq_soft).sum() * w.shape[-1]
            loss = (ce_loss - LAMBDA_ENT * entropy
                    + LAMBDA_MEM * aux_memory + LAMBDA_ORG * aux_organelle)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step_global += 1
            e_sum += loss.item()
            e_n += 1
            del logits, loss
        guard.restore()
        print(f"[{tag} EPOCH {epoch}/{EPOCHS}] train_loss={e_sum / e_n:.4f} "
              f"({(time.time() - t0):.0f}s)")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return e_sum / e_n, time.time() - t0


def load_base():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    cfg = ckpt['cfg']
    model = HierarchicalCellMoE_LM(d=cfg['d'], vocab_size=cfg['vocab_size'],
                                   n_cells=cfg['n_cells'],
                                   n_organelles=cfg['n_organelles'],
                                   n_memory=cfg['n_memory'],
                                   topk_organelle=cfg['topk_organelle'],
                                   topk_cell=cfg['topk_cell'],
                                   L=cfg['L']).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    return model, cfg


def main():
    set_seed(SEED)
    vocab_size, train_ids, val_ids, _ = load_shakespeare()

    model, cfg = load_base()
    print(f"[3.1] cfg={cfg}")
    _, base_ppl = eval_ppl(model, val_ids, vocab_size, SEQ_LEN, n_batch=20, batch_size=16)
    print(f"[3.1] 基线 shakespeare_ppl={base_ppl:.3f} (ckpt记录={ckpt_final_ppl():.3f})")
    del _
    rows = [("baseline", round(base_ppl, 4), "n/a")]

    # ---- 3.2 抹除 50% ----
    erase_memories(model, 0.5)
    _, ppl50 = eval_ppl(model, val_ids, vocab_size, SEQ_LEN, n_batch=20, batch_size=16)
    print(f"[3.2] erased_50 shakespeare_ppl={ppl50:.3f}")
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                "erase_ratio": 0.5, "base_ppl": base_ppl}, ERASE50_CKPT)
    rows.append(("erased_50", round(ppl50, 4), "n/a"))
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 3.3 抹除 100%（基于新加载的干净基线） ----
    model2, _ = load_base()
    erase_memories(model2, 1.0)
    _, ppl100 = eval_ppl(model2, val_ids, vocab_size, SEQ_LEN, n_batch=20, batch_size=16)
    print(f"[3.3] erased_100 shakespeare_ppl={ppl100:.3f}")
    torch.save({"state_dict": model2.state_dict(), "cfg": cfg,
                "erase_ratio": 1.0, "base_ppl": base_ppl}, ERASE100_CKPT)
    rows.append(("erased_100", round(ppl100, 4), "n/a"))
    del model2
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 3.4 重训恢复（50% 与 100% 各 5ep） ----
    # 50% 恢复
    model50, _ = load_base()
    erase_memories(model50, 0.5)
    idx50 = build_erase_index(model50, 0.5)
    guard50 = FrozenMemoryGuard(model50, collect_keep_values(model50, idx50))
    _, _ = train_recover(model50, train_ids, vocab_size, guard50, "recover50")
    _, rec50_ppl = eval_ppl(model50, val_ids, vocab_size, SEQ_LEN, n_batch=20, batch_size=16)
    rec50 = "yes" if rec50_ppl - base_ppl < 1.0 else "no"
    print(f"[3.4] recover_50 shakespeare_ppl={rec50_ppl:.3f} recoverable={rec50}")
    rows.append(("recover_50", round(rec50_ppl, 4), rec50))
    del model50
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # 100% 恢复
    model100, _ = load_base()
    erase_memories(model100, 1.0)
    idx100 = build_erase_index(model100, 1.0)
    guard100 = FrozenMemoryGuard(model100, collect_keep_values(model100, idx100))
    _, _ = train_recover(model100, train_ids, vocab_size, guard100, "recover100")
    _, rec100_ppl = eval_ppl(model100, val_ids, vocab_size, SEQ_LEN, n_batch=20, batch_size=16)
    rec100 = "yes" if rec100_ppl - base_ppl < 1.0 else "no"
    print(f"[3.4] recover_100 shakespeare_ppl={rec100_ppl:.3f} recoverable={rec100}")
    rows.append(("recover_100", round(rec100_ppl, 4), rec100))
    del model100
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 3.5 输出 CSV ----
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "shakespeare_ppl", "recoverable"])
        w.writerows(rows)
    print(f"[3.5] CSV -> {OUT_CSV}")
    print("[DONE] 任务3 完成")


def ckpt_final_ppl():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    return ckpt.get("final_ppl", float("nan"))


if __name__ == "__main__":
    main()
