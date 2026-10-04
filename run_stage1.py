# -*- coding: utf-8 -*-
"""阶段一：验证 improved 版本不破坏原有能力（v3 完整训练）
- 模型：ImprovedHierarchicalCellMoE_LM（4 细胞 x 6 细胞器 x 24 忆点）
- 数据：莎士比亚 90/10（与旧版 v3 相同数据管线）
- seed=2026, 5 epoch, batch=128, seq_len=64, lr=3e-4, warmup 5%, clip 1.0
- 机制正则沿用 v3（LAMBDA_ENT/MEM/ORG=0.05，不调 lambda）
- 保存 checkpoint -> checkpoints/improved_v3_2026.pt
- 报告 PPL，与旧版 v3 的 12.78 对比，验收 PPL<=13.30，>13.5 停
"""
import csv
import gc
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from snowflake_moe_improved import ImprovedHierarchicalCellMoE_LM  # noqa: E402
from train_lm import load_shakespeare, lm_batch  # noqa: E402

SEQ_LEN, BATCH_SIZE, EPOCHS, LR, SEED = 64, 128, 5, 3e-4, 2026
LAMBDA_ENT, LAMBDA_MEM, LAMBDA_ORG = 0.05, 0.05, 0.05
CKPT_PATH = os.path.join(BASE, "checkpoints", "improved_v3_2026.pt")
OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output", "results_stage1.csv"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_full(model, ids, vocab_size, batch_size=16):
    """确定性全量评估：连续分块覆盖全部 token，返回 (avg_ce, ppl)。"""
    model.eval()
    k = batch_size * SEQ_LEN
    n = ids.numel()
    total_ce, total_n = 0.0, 0
    with torch.no_grad():
        s = 0
        while s + k + 1 <= n:
            x = ids[s:s + k].view(batch_size, SEQ_LEN).to(DEVICE)
            y = ids[s + 1:s + k + 1].view(batch_size, SEQ_LEN).to(DEVICE)
            logits, _ = model(x)
            ce = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
            total_ce += ce * k
            total_n += k
            s += k
    model.train()
    avg_ce = total_ce / total_n
    return avg_ce, math.exp(avg_ce)


def main():
    set_seed(SEED)
    vocab_size, train_ids, val_ids, chars = load_shakespeare()
    print(f"[S1] vocab={vocab_size} train={train_ids.numel()} val={val_ids.numel()}")

    model = ImprovedHierarchicalCellMoE_LM(
        d=64, vocab_size=vocab_size, n_cells=4, n_organelles=6, n_memory=24,
        topk_organelle=4, topk_cell=2, L=2, gate_ent_reg=True).to(DEVICE)
    print(f"[S1] ImprovedHierarchicalCellMoE_LM params={count_params(model)}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_steps = max(1, train_ids.numel() // (SEQ_LEN * BATCH_SIZE))
    total_steps = n_steps * EPOCHS
    warmup = max(1, int(total_steps * 0.05))

    def lr_at(step):
        return LR * (step + 1) / warmup if step < warmup else LR

    t0 = time.time()
    step_global = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        e_sum, e_n = 0.0, 0
        for _ in range(n_steps):
            xb, yb = lm_batch(train_ids, BATCH_SIZE, SEQ_LEN, seed=SEED + step_global)
            for g in opt.param_groups:
                g['lr'] = lr_at(step_global)
            opt.zero_grad(set_to_none=True)
            logits, info = model(xb)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            loss = ce_loss
            if isinstance(info, dict):
                # 层级改进版：info = {gate, gate_entropy, cell_infos}
                cell_infos = info.get('cell_infos', [info])
                ent_sum = mem_sum = org_sum = 0.0
                for ci in cell_infos:
                    w = ci['weights']
                    ent_sum += -(w * torch.log(w + 1e-9)).sum(-1).mean()
                    memory_usage = ci['memory_attn'].mean(dim=0)
                    mem_sum += (memory_usage * memory_usage).sum() * memory_usage.shape[-1]
                    freq_soft = w.mean(dim=0)
                    org_sum += (freq_soft * freq_soft).sum() * w.shape[-1]
                nc = len(cell_infos)
                # 顶层 gate 熵（可导）
                gate = info['gate']
                gate_ent = -(gate * (gate + 1e-9).log()).sum(-1).mean()
                loss = (ce_loss - LAMBDA_ENT * (ent_sum / nc + 0.25 * gate_ent)
                        + LAMBDA_MEM * (mem_sum / nc) + LAMBDA_ORG * (org_sum / nc))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step_global += 1
            e_sum += loss.item()
            e_n += 1
            del logits, loss
        print(f"[S1 EPOCH {epoch}/{EPOCHS}] train_loss={e_sum / e_n:.4f} "
              f"({(time.time() - t0):.0f}s)", flush=True)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # 全量评估 val
    val_ce, val_ppl = eval_full(model, val_ids, vocab_size)
    print(f"[S1] FINAL val_ce={val_ce:.4f} val_ppl={val_ppl:.4f}")

    # 保存 checkpoint
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": dict(d=64, vocab_size=vocab_size, n_cells=4, n_organelles=6,
                       n_memory=24, topk_organelle=4, topk_cell=2, L=2,
                       gate_ent_reg=True),
        "val_ppl": val_ppl, "val_ce": val_ce, "seed": SEED,
        "epochs": EPOCHS, "vocab": chars,
    }, CKPT_PATH)
    print(f"[S1] CKPT -> {CKPT_PATH}")

    # 输出 CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "model", "params", "val_ce", "val_ppl", "old_v3_ppl", "sec"])
        w.writerow(["1", "improved_hier_4c6o24m", count_params(model),
                    round(val_ce, 4), round(val_ppl, 4), 12.78,
                    round(time.time() - t0, 1)])
    print(f"[S1] CSV -> {OUT_CSV}")

    # 验收判定
    if val_ppl > 13.5:
        print(f"[S1] FAIL: PPL={val_ppl:.4f} > 13.5，改造引入退化，停止后续阶段")
    else:
        print(f"[S1] PASS: PPL={val_ppl:.4f} <= 13.30 验收线"
              + ("" if val_ppl <= 13.30 else "（注意：>13.30 但 <=13.5，不差于旧版线内需人工判定）"))


if __name__ == "__main__":
    main()
