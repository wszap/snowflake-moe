# -*- coding: utf-8 -*-
"""消融验证：置零 mem_gate（route_proj.weight）对比 PPL。
验收第三指标：消融置零后 PPL 变差 => mem_gate 对路由有真实贡献。
用法：训练出最终 checkpoint（cellmoe_tinystories.pt）后运行。
"""
import os
import sys
import math

import torch

import run_cellmoe_ckpt as R


def collect_cells(model):
    """递归收集所有 FastCellMoE 实例。"""
    cells = []
    for layer in model.layers:                      # FastHierLM.layers
        for cell in layer.cells:                    # FastHierCellMoE.cells
            cells.append(cell)
    return cells


def main():
    if not os.path.exists(R.CKPT_PATH):
        print(f"[ABL] 未找到最终 checkpoint: {R.CKPT_PATH}，训练可能未完成")
        sys.exit(1)

    ckpt = torch.load(R.CKPT_PATH, map_location=R.DEVICE)
    cfg = ckpt["cfg"]
    print(f"[ABL] 加载 checkpoint: final_ppl={ckpt.get('final_ppl', '?')}")

    vocab_size, train_ids, val_ids, chars = R.load_tinystories()
    val_ids = val_ids.to(R.DEVICE)

    model = R.FastHierLM(
        d=cfg["d"], vocab_size=cfg["vocab_size"], n_cells=cfg["n_cells"],
        n_organelles=cfg["n_organelles"], n_memory=cfg["n_memory"],
        topk_organelle=cfg["topk_organelle"], topk_cell=cfg["topk_cell"],
        L=cfg["L"]).to(R.DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 1) 原版 PPL
    ce0, ppl0 = R.eval_full(model, val_ids, vocab_size)
    print(f"[ABL] 原版        val_ce={ce0:.4f} PPL={ppl0:.4f}")

    # 2) 置零 mem_gate（route_proj.weight -> 0，退化为纯 assembly 路由）
    cells = collect_cells(model)
    for c in cells:
        c.route_proj.weight.data.zero_()
    ce1, ppl1 = R.eval_full(model, val_ids, vocab_size)
    print(f"[ABL] 置零mem_gate val_ce={ce1:.4f} PPL={ppl1:.4f}")

    d_ppl = ppl1 - ppl0
    verdict = "通过：mem_gate 有贡献（PPL 变差）" if d_ppl > 0 else "未通过：置零后 PPL 未变差"
    print(f"[ABL] ΔPPL={d_ppl:+.4f} -> {verdict}")


if __name__ == "__main__":
    main()
