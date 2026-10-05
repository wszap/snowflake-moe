# -*- coding: utf-8 -*-
"""阶段三：分组 softmax 防稀释验证（improved 版）

流程：
1. 加载 checkpoints/improved_v3_2026.pt（ImprovedHierarchicalCellMoE_LM）
2. 每个细胞 add_new_memory(8)，冻结全部旧参数，只解冻新忆点
3. 用新领域（KJV Bible，同一字符表）训练 3 epoch
4. 评估：旧领域（莎士比亚 val）PPL + 新领域（圣经 test）PPL
验收：旧领域 PPL 退化 < 0.3；新领域 PPL 下降 > 20%
输出 output/results_lifelong_improved.csv
红线：不改架构、不调 lambda、empty_cache + gc.collect、GPU 温度 < 80
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

CKPT_PATH = os.path.join(BASE, "checkpoints", "improved_v3_2026.pt")
LIFELONG_CKPT = os.path.join(BASE, "checkpoints", f"improved_v3_lifelong_{SEED}.pt")
DATA_DIR = os.path.join(BASE, "data")
NEW_TRAIN = os.path.join(DATA_DIR, "new_domain_train.pt")
NEW_TEST = os.path.join(DATA_DIR, "new_domain_test.pt")
OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output",
                                       "results_lifelong_improved.csv"))

SEQ_LEN, BATCH_SIZE, EPOCHS, LR, SEED = 64, 128, 3, 1e-3, 2026
LAMBDA_ENT, LAMBDA_MEM, LAMBDA_ORG = 0.05, 0.05, 0.05
NEW_ADD = 8          # 每个细胞新增忆点数


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


def new_memory_utilization(model, ids, vocab_size, n_memory=24,
                           n_batch=20, batch_size=16):
    """路由测试：新领域输入上，忆点检索 argmax 落在新忆点区间(>=n_memory)的比例。"""
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for _ in range(n_batch):
            xb, _ = lm_batch(ids, batch_size, SEQ_LEN)
            _, info = model(xb)
            for ci in info['cell_infos']:
                am = ci['memory_attn']                 # [N, n_memory+add]
                hits += int((am.argmax(-1) >= n_memory).sum().item())
                total += am.shape[0]
    model.train()
    return hits / max(1, total)


def main():
    set_seed(SEED)
    vocab_size, train_ids, val_ids, chars = load_shakespeare()

    # ---- 新领域数据 ----
    if not (os.path.exists(NEW_TRAIN) and os.path.exists(NEW_TEST)):
        raise FileNotFoundError(f"缺少圣经数据 {NEW_TRAIN}/{NEW_TEST}，先跑 prepare")
    new_train = torch.load(NEW_TRAIN)
    new_test = torch.load(NEW_TEST)
    print(f"[S3] vocab={vocab_size} shk_train={train_ids.numel()} "
          f"shk_val={val_ids.numel()} bible_train={new_train.numel()} "
          f"bible_test={new_test.numel()}")

    # ---- 加载 improved ckpt ----
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    cfg = ckpt['config']
    model = ImprovedHierarchicalCellMoE_LM(
        d=cfg['d'], vocab_size=cfg['vocab_size'], n_cells=cfg['n_cells'],
        n_organelles=cfg['n_organelles'], n_memory=cfg['n_memory'],
        topk_organelle=cfg['topk_organelle'], topk_cell=cfg['topk_cell'],
        L=cfg['L'], gate_ent_reg=cfg.get('gate_ent_reg', True)).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    print(f"[S3] ckpt loaded, recorded val_ppl={ckpt['val_ppl']:.4f}")

    # ---- 基线评估 ----
    _, base_ppl = eval_full(model, val_ids, vocab_size)
    _, base_new_ppl = eval_full(model, new_test, vocab_size)
    print(f"[S3 BASELINE] shk_ppl={base_ppl:.4f} bible_ppl={base_new_ppl:.4f}")

    # ---- 加新忆点 + 冻结旧参数 ----
    n_cells = 0
    for layer in model.layers:
        for cell in layer.cells:
            cell.add_new_memory(NEW_ADD, seed=SEED)
            n_cells += 1
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in model.layers:
        for cell in layer.cells:
            cell.new_memory_keys.requires_grad_(True)
            cell.new_memory_assembly.requires_grad_(True)
            if cell.new_memory_value is not None:
                cell.new_memory_value.requires_grad_(True)
    trainable = count_params(model)
    print(f"[S3] 冻结完成：{n_cells} 个细胞各 +{NEW_ADD}，可训练参数={trainable}（新忆点 keys/asm/val）")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 只训练新忆点：圣经 3 epoch ----
    t0 = time.time()
    # keys/asm 正常学习（wd 压范数防旧领域误激活）；val 因 memory_read_scale=25 梯度放大 25 倍，
    # 用极小 lr+强 wd 平衡到可感知但不膨胀的量级；group_w 冻结（旧忆点权重基线 0.62）
    key_params = [p for name, p in model.named_parameters()
                  if p.requires_grad and ("new_memory_keys" in name
                                          or "new_memory_assembly" in name)]
    val_params = [p for name, p in model.named_parameters()
                  if p.requires_grad and "new_memory_value" in name]
    opt = torch.optim.Adam([
        {"params": key_params, "lr": LR, "weight_decay": 1e-2},
        {"params": val_params, "lr": 3e-6, "weight_decay": 1e-1},
    ])
    n_steps = max(1, new_train.numel() // (SEQ_LEN * BATCH_SIZE))
    step_global = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        e_sum, e_n = 0.0, 0
        for _ in range(n_steps):
            xb, yb = lm_batch(new_train, BATCH_SIZE, SEQ_LEN, seed=SEED + step_global)
            opt.zero_grad(set_to_none=True)
            logits, info = model(xb)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
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
            gate = info['gate']
            gate_ent = -(gate * (gate + 1e-9).log()).sum(-1).mean()
            loss = (ce_loss - LAMBDA_ENT * (ent_sum / nc + 0.25 * gate_ent)
                    + LAMBDA_MEM * (mem_sum / nc) + LAMBDA_ORG * (org_sum / nc))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step_global += 1
            e_sum += loss.item()
            e_n += 1
            del logits, loss
        print(f"[S3 EPOCH {epoch}/{EPOCHS}] train_loss={e_sum / e_n:.4f} "
              f"({(time.time() - t0):.0f}s)", flush=True)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    os.makedirs(os.path.dirname(LIFELONG_CKPT), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg,
                "base_shk_ppl": base_ppl, "base_bible_ppl": base_new_ppl,
                "new_add": NEW_ADD}, LIFELONG_CKPT)
    print(f"[S3] SAVED -> {LIFELONG_CKPT}")

    # ---- 训练后评估 ----
    _, life_ppl = eval_full(model, val_ids, vocab_size)
    _, life_new_ppl = eval_full(model, new_test, vocab_size)
    util = new_memory_utilization(model, new_test, vocab_size,
                                  n_memory=cfg['n_memory'])
    regress = life_ppl - base_ppl
    drop = (base_new_ppl - life_new_ppl) / max(1e-9, base_new_ppl)
    print(f"[S3 FINAL] shk_ppl={life_ppl:.4f} (base {base_ppl:.4f}, 退化{regress:+.4f})  "
          f"bible_ppl={life_new_ppl:.4f} (base {base_new_ppl:.4f}, 下降{drop:.2%})  "
          f"new_mem_util={util:.3%}")

    # ---- CSV ----
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "shk_base_ppl", "shk_life_ppl", "shk_regress",
                    "bible_base_ppl", "bible_life_ppl", "bible_drop",
                    "new_mem_util", "pass"])
        w.writerow(["3", round(base_ppl, 4), round(life_ppl, 4), round(regress, 4),
                    round(base_new_ppl, 4), round(life_new_ppl, 4), round(drop, 4),
                    round(util, 4), 1 if (regress < 0.3 and drop > 0.2) else 0])
    print(f"[S3] CSV -> {OUT_CSV}")

    # ---- 验收 ----
    ok_regress = regress < 0.3
    ok_drop = drop > 0.20
    if ok_regress and ok_drop:
        print(f"[S3 PASS] 旧领域退化{regress:.4f}<0.3 且 新领域下降{drop:.2%}>20%")
    else:
        print(f"[S3 FAIL] 退化{regress:.4f}('+'){'' if ok_regress else ' 不达标'} / "
              f"下降{drop:.2%}{'' if ok_drop else ' 不达标'}")


if __name__ == "__main__":
    main()
