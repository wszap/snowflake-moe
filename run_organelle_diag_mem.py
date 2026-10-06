# -*- coding: utf-8 -*-
"""锁10诊断补跑：mem_gate 消融 + 完整汇总。

organelle 消融数据（128 条）从 organelle_diag_lock10.log 解析（该轮已跑完但汇总处
因 bug 中断）；本脚本补跑 mem_gate 消融（route_proj 置零）并给出细胞器 vs 忆点对比。
"""
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_cellmoe_ckpt as m

CKPT = os.path.join(m.BASE, "checkpoints", "cellmoe_tinystories.pt")


def main():
    # ---- 1. 解析日志中的 organelle ΔPPL ----
    log_path = os.path.join(m.BASE, "output", "organelle_diag_lock10.log")
    pat = re.compile(r"L(\d+) C(\d+) org(\d+): dPPL=([+-][\d.]+)")
    deltas = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            mm = pat.search(line)
            if mm:
                deltas.append((int(mm.group(1)), int(mm.group(2)),
                               int(mm.group(3)), float(mm.group(4))))
    d_all = [d[3] for d in deltas]
    d_mean = sum(d_all) / len(d_all) if d_all else 0.0
    d_max = max(deltas, key=lambda t: abs(t[3]))
    print(f"[DIAG] parsed {len(deltas)} organelle ablations: "
          f"mean_dPPL={d_mean:+.4f} max={d_max[3]:+.4f} "
          f"@L{d_max[0]}C{d_max[1]}org{d_max[2]}", flush=True)

    # ---- 2. 加载模型 + 文档边界 val_ids ----
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

    # ---- 3. mem_gate 消融：route_proj 置零（忆点不参与路由）----
    n_l, n_c = cfg["L"], cfg["n_cells"]
    mem_deltas = []
    for li in range(n_l):
        for ci in range(n_c):
            cell = model.layers[li].cells[ci]
            rp_orig = cell.route_proj.weight.data.clone()
            cell.route_proj.weight.data.zero_()
            _, ppl = m.eval_full(model, val_ids, vocab_size)
            d = ppl - base_ppl
            mem_deltas.append(d)
            cell.route_proj.weight.data = rp_orig
            print(f"[DIAG] L{li} C{ci} mem_gate off: dPPL={d:+.4f}", flush=True)
    mem_mean = sum(mem_deltas) / len(mem_deltas)
    print(f"[DIAG] mem_gate ablation: mean_dPPL={mem_mean:+.4f}", flush=True)

    # ---- 4. 结论 ----
    if abs(mem_mean) > 1e-9:
        ratio = d_mean / mem_mean
        print(f"[DIAG] CONCLUSION: organelle mean={d_mean:+.4f} vs "
              f"mem_gate mean={mem_mean:+.4f} -> organelle/mem ratio={ratio:.2f}x",
              flush=True)
        if d_mean >= 0.3:
            print("[DIAG] organelle 单个消融影响显著(>=0.3)：细胞器重要，"
                  "忆点(≈0.07)为次要调节", flush=True)
        elif d_mean <= 0.1:
            print("[DIAG] organelle 单个消融影响微弱(<=0.1)：整个 MoE 层未强分化，"
                  "问题在 router 训练（忆点非主因）", flush=True)
        else:
            print("[DIAG] organelle 单个消融影响中等(0.1~0.3)", flush=True)
    else:
        print("[DIAG] mem_gate 无贡献，无法比较", flush=True)


if __name__ == "__main__":
    main()
