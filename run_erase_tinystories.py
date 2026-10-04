# -*- coding: utf-8 -*-
"""任务1：可擦除性验证（CellMoE TinyStories checkpoint）
- 加载 temp/checkpoints/cellmoe_tinystories.pt（run_cellmoe_ckpt.py 训练，
  超参与 results_tinystories.csv 完全一致：d128 L4 4cells 8org 32mem 25MB 5ep）
- 基线 val PPL；抹除 50% memory_value；抹除 100% memory_value
- 输出 output/results_erase_tinystories.csv
- 验收：100% 擦除 PPL>30 成立；若 <15 需放大 memory_read 系数重训
不改架构、不调 lambda；复用 run_cellmoe_ckpt.py 的模型类与 eval 逻辑。
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
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from run_cellmoe_ckpt import (  # noqa: E402
    DEVICE, SEQ_LEN, FastHierLM, count_params, eval_full,
    load_tinystories, set_seed,
)

CKPT_PATH = os.path.join(BASE, "checkpoints", "cellmoe_tinystories.pt")
OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output",
                                       "results_erase_tinystories.csv"))
SEED = 2026


def erase_memory_value(model, frac, seed=SEED):
    """按 memory slot 抹除 memory_value：frac=0.5/1.0。
    每个 memory_value 参数形状 (n_memory, d)；抹除即将该 slot 的
    value 行整体置零（记忆内容清零，架构/门控不变）。
    返回抹除的 slot 总数。
    """
    rng = random.Random(seed + int(round(frac * 1000)))
    total = 0
    names = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "memory_value" in name:
                names.append(name)
                n_slots = p.shape[0]
                k = int(round(n_slots * frac))
                idx = sorted(rng.sample(range(n_slots), k))
                p.data[idx] = 0.0
                total += len(idx)
    return total, names


def main():
    set_seed(SEED)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    cfg = ckpt["cfg"]
    print(f"[Erase] ckpt final_ppl={ckpt['final_ppl']:.4f} cfg={cfg}")
    vocab_size, train_ids, val_ids, chars = load_tinystories()

    rows = []
    # ---- 基线 ----
    model = FastHierLM(d=cfg["d"], vocab_size=cfg["vocab_size"],
                       n_cells=cfg["n_cells"], n_organelles=cfg["n_organelles"],
                       n_memory=cfg["n_memory"],
                       topk_organelle=cfg["topk_organelle"],
                       topk_cell=cfg["topk_cell"], L=cfg["L"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    n_params = count_params(model)
    ce, ppl = eval_full(model, val_ids, vocab_size)
    print(f"[Erase] baseline val_ce={ce:.4f} val_ppl={ppl:.4f} params={n_params}")
    rows.append(["CellMoE", n_params, 0.0, round(ce, 4), round(ppl, 4),
                 f"baseline (ckpt final_ppl={ckpt['final_ppl']:.4f})"])
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    for frac in (0.5, 1.0):
        model = FastHierLM(d=cfg["d"], vocab_size=cfg["vocab_size"],
                           n_cells=cfg["n_cells"],
                           n_organelles=cfg["n_organelles"],
                           n_memory=cfg["n_memory"],
                           topk_organelle=cfg["topk_organelle"],
                           topk_cell=cfg["topk_cell"], L=cfg["L"]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        n_slots, names = erase_memory_value(model, frac, seed=SEED)
        ce, ppl = eval_full(model, val_ids, vocab_size)
        print(f"[Erase] erase {frac*100:.0f}% ({n_slots} slots) "
              f"val_ce={ce:.4f} val_ppl={ppl:.4f}")
        rows.append(["CellMoE", n_params, frac, round(ce, 4), round(ppl, 4),
                     f"erased {frac*100:.0f}% memory_value "
                     f"({n_slots} slots/{len(names)} tensors)"])
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "erase_frac", "val_ce", "val_ppl",
                    "note"])
        w.writerows(rows)
    print(f"[Erase] CSV -> {OUT_CSV}")

    base_ppl = rows[0][4]
    erase100_ppl = rows[2][4]
    print(f"[Erase] 验收: 100% 擦除 PPL={erase100_ppl:.4f} "
          f"(>30 成立, <15 需放大 memory_read 重训)")


if __name__ == "__main__":
    main()
