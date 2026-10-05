# -*- coding: utf-8 -*-
"""阶段二：验证 memory_value 安全机制
- 加载 checkpoints/improved_v3_2026.pt（ImprovedHierarchicalCellMoE_LM 4x6x24）
- 测基线 PPL + gate 熵
- 抹除 50% memory_value（置零）-> 测 PPL + gate 熵
- 抹除 100% memory_value（置零）-> 测 PPL + gate 熵
- 输出 output/results_erase_improved.csv
- 验收：100% 擦除后 PPL > 30；< 20 则需放大 memory_read 系数
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
from train_lm import load_shakespeare  # noqa: E402

SEQ_LEN = 64
CKPT_PATH = os.path.join(BASE, "checkpoints", "improved_v3_2026.pt")
OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output",
                                       "results_erase_improved.csv"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def build_model(vocab_size, ckpt):
    model = ImprovedHierarchicalCellMoE_LM(
        d=64, vocab_size=vocab_size, n_cells=4, n_organelles=6, n_memory=24,
        topk_organelle=4, topk_cell=2, L=2, gate_ent_reg=True).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def erase_memory_value(model, ratio, seed=2026):
    """按固定 seed 随机选择 memory_value 行置零，返回被置零的 (模块, 行idx) 列表。"""
    rng = random.Random(seed)
    erased = []
    for name, p in model.named_parameters():
        if name.endswith("memory_value"):
            n = p.shape[0]
            k = int(round(n * ratio))
            idx = sorted(rng.sample(range(n), k))
            with torch.no_grad():
                p.data[idx, :] = 0.0
            erased.append((name, idx))
    return erased


def eval_full_info(model, ids, vocab_size, batch_size=16):
    """确定性全量评估，返回 (avg_ce, ppl, gate_entropy_mean)。"""
    model.eval()
    k = batch_size * SEQ_LEN
    n = ids.numel()
    total_ce, total_n = 0.0, 0
    ent_sum, ent_n = 0.0, 0
    with torch.no_grad():
        s = 0
        while s + k + 1 <= n:
            x = ids[s:s + k].view(batch_size, SEQ_LEN).to(DEVICE)
            y = ids[s + 1:s + k + 1].view(batch_size, SEQ_LEN).to(DEVICE)
            logits, info = model(x)
            ce = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
            total_ce += ce * k
            total_n += k
            if isinstance(info, dict) and info.get("gate_entropy") is not None:
                ent_sum += info["gate_entropy"]
                ent_n += 1
            s += k
    model.train()
    avg_ce = total_ce / total_n
    avg_ent = ent_sum / ent_n if ent_n else float("nan")
    return avg_ce, math.exp(avg_ce), avg_ent


def main():
    set_seed(2026)
    vocab_size, train_ids, val_ids, chars = load_shakespeare()
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    print(f"[S2] ckpt loaded, val={val_ids.numel()}")

    rows = []
    # ---- 基线 ----
    model = build_model(vocab_size, ckpt)
    ce0, ppl0, ent0 = eval_full_info(model, val_ids, vocab_size)
    print(f"[S2 BASELINE] val_ce={ce0:.4f} val_ppl={ppl0:.4f} gate_entropy={ent0:.4f}")
    rows.append(("baseline", 0.0, round(ppl0, 4), round(ent0, 4)))
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 抹除 50% ----
    model = build_model(vocab_size, ckpt)
    erased50 = erase_memory_value(model, 0.5)
    ce50, ppl50, ent50 = eval_full_info(model, val_ids, vocab_size)
    print(f"[S2 ERASE50] val_ce={ce50:.4f} val_ppl={ppl50:.4f} gate_entropy={ent50:.4f} "
          f"({len(erased50)} 组 {sum(len(v) for _, v in erased50)} 行置零)")
    rows.append(("erase_50", 0.5, round(ppl50, 4), round(ent50, 4)))
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 抹除 100% ----
    model = build_model(vocab_size, ckpt)
    erased100 = erase_memory_value(model, 1.0)
    ce100, ppl100, ent100 = eval_full_info(model, val_ids, vocab_size)
    print(f"[S2 ERASE100] val_ce={ce100:.4f} val_ppl={ppl100:.4f} gate_entropy={ent100:.4f} "
          f"({len(erased100)} 组 {sum(len(v) for _, v in erased100)} 行置零)")
    rows.append(("erase_100", 1.0, round(ppl100, 4), round(ent100, 4)))
    # ---- checkpoint：最终模型保存（100% 擦除后，不改训练逻辑）----
    ckpt_path = os.path.join(BASE, "checkpoints", "stage2_2026.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "cfg": ckpt.get("cfg", {}),
                "erase_ratio": 1.0}, ckpt_path)
    print(f"[S2] CKPT saved -> {ckpt_path}", flush=True)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 输出 CSV ----
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "erase_ratio", "val_ppl", "gate_entropy"])
        w.writerows(rows)
    print(f"[S2] CSV -> {OUT_CSV}")

    # ---- 验收 ----
    if ppl100 > 30:
        print(f"[S2 PASS] 100% 擦除 PPL={ppl100:.4f} > 30，忆点确为记忆载体")
    elif ppl100 < 20:
        print(f"[S2 ACTION] 100% 擦除 PPL={ppl100:.4f} < 20，memory_value 权重太小，"
              f"需在 forward 放大 memory_read 系数后重跑")
    else:
        print(f"[S2 INFO] 100% 擦除 PPL={ppl100:.4f} 落在 20~30，按数据判定")


if __name__ == "__main__":
    main()
