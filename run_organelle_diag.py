# -*- coding: utf-8 -*-
"""锁10后诊断：对比细胞器 vs 忆点 相对贡献。

对刚训练完成的 cellmoe_tinystories.pt：
- base：整体 val PPL
- 关掉单个 organelle（W1[i] 置零）：per (layer, cell, organelle) ΔPPL
- 关掉 mem_gate（route_proj.weight 置零，忆点不参与路由）：整体 ΔPPL

决策：若单细胞器 ΔPPL ≈ 0.3 → 细胞器重要，忆点(≈0.07)次要；
若单细胞器也只涨 ~0.05 → 整个 MoE 层未形成强分化，问题在 router 训练。
"""
import gc
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_cellmoe_ckpt as m

CKPT = os.path.join(m.BASE, "checkpoints", "cellmoe_tinystories.pt")


def main():
    ckpt = torch.load(CKPT, map_location="cpu")
    cfg = ckpt["cfg"]
    vocab_size = cfg["vocab_size"]
    model = m.FastHierLM(d=cfg["d"], vocab_size=vocab_size,
                         n_cells=cfg["n_cells"], n_organelles=cfg["n_organelles"],
                         n_memory=cfg["n_memory"],
                         topk_organelle=cfg["topk_organelle"],
                         topk_cell=cfg["topk_cell"], L=cfg["L"]).to(m.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, _, val_ids, _ = m.load_tinystories()
    base_ce, base_ppl = m.eval_full(model, val_ids, vocab_size)
    print(f"[DIAG] base val_ppl={base_ppl:.4f} (ce={base_ce:.4f})", flush=True)

    # ---- 关掉单个 organelle：W1[i] 置零 ----
    deltas = []
    n_l, n_c, n_o = cfg["L"], cfg["n_cells"], cfg["n_organelles"]
    for li in range(n_l):
        for ci in range(n_c):
            cell = model.layers[li].cells[ci]
            for oi in range(n_o):
                w1_orig = cell.W1.data[oi].clone()
                cell.W1.data[oi].zero_()
                _, ppl = m.eval_full(model, val_ids, vocab_size)
                delta = ppl - base_ppl
                deltas.append((li, ci, oi, delta))
                cell.W1.data[oi] = w1_orig
                print(f"[DIAG] L{li} C{ci} org{oi}: dPPL={delta:+.4f}", flush=True)
                gc.collect()
                if m.DEVICE == "cuda":
                    torch.cuda.empty_cache()

    d_all = [d for _, _, _, d in deltas]
    d_mean = sum(d_all) / len(d_all)
    d_max = max(deltas, key=lambda t: abs(t[3]))
    print(f"[DIAG] organelle ablation: mean_dPPL={d_mean:+.4f} "
          f"max={d_max[3]:+.4f} @L{d_max[0]}C{d_max[1]}org{d_max[2]}", flush=True)

    # ---- 关掉 mem_gate：route_proj 置零（忆点不参与路由）----
    mem_deltas = []
    for li in range(n_l):
        for ci in range(n_c):
            cell = model.layers[li].cells[ci]
            rp_orig = cell.route_proj.weight.data.clone()
            cell.route_proj.weight.data.zero_()
            _, ppl = m.eval_full(model, val_ids, vocab_size)
            mem_deltas.append(ppl - base_ppl)
            cell.route_proj.weight.data = rp_orig
            print(f"[DIAG] L{li} C{ci} mem_gate off: dPPL={ppl - base_ppl:+.4f}",
                  flush=True)
            gc.collect()
            if m.DEVICE == "cuda":
                torch.cuda.empty_cache()
    mem_mean = sum(mem_deltas) / len(mem_deltas)
    print(f"[DIAG] mem_gate ablation: mean_dPPL={mem_mean:+.4f}", flush=True)
    print(f"[DIAG] CONCLUSION: organelle mean={d_mean:+.4f} vs mem_gate mean="
          f"{mem_mean:+.4f} -> organelle/mem ratio="
          f"{d_mean / mem_mean:.2f}x" if mem_mean != 0 else
          "[DIAG] mem_gate 无贡献，无法比较", flush=True)


if __name__ == "__main__":
    main()
