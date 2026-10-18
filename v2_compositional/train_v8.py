# -*- coding: utf-8 -*-
"""v8 训练脚本：真·拼专家。

与 train_v4.py 完全同源（复用 load_data / evaluate / leak_check），
唯一区别是把细胞器换成 ComposableExpert。
数据管线、评估口径、H2 判据全部不变，保证与 v3/v4 可直接对比。

新增参数：
  --k        每个 chunk 自主选出几个忆点来拼（拼装粒度）
  --h-cell   每个忆点的独立隐层宽度（拼后等效宽度 = k × h_cell）
  --chunk    几个 token 共用一组专家（显存安全阀，见下）

显存说明（重要）：
  E1[topi] 的形状是 [C, k, d, h_cell]，C 是 chunk 数。
  chunk=1 时 C=B*T，会炸显存；chunk=32 时 C=B*T/32，可控。
  所以默认 chunk=32 —— 同一段文本共用一组专家，这也正是原设计的语义。
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from train_v3 import load_data, evaluate
from scaffold_v3 import leak_check
from snowflake_v4 import SnowflakeLMV4
from composable_expert import ComposableExpert


def build_v8(a, vocab_size, device):
    """先按 v4 搭好骨架（含核架/嵌入/位置/norm），再把细胞器换成真拼专家。"""
    model = SnowflakeLMV4(
        vocab_size=vocab_size, d=a.d, n=a.n, L=a.L, h=a.h,
        n_cells=a.n_cells, seq_len=a.seq_len, rank=a.rank, temp=a.temp,
        n_head=a.n_head, scaffold_top_k=a.scaffold_top_k,
        select_mode=a.select_mode,
        variant="base",          # 共享基底已内置在 ComposableExpert，不再外接
        shared_mult=a.shared_mult, seed=a.seed,
    ).to(device)

    for li in range(a.L):
        for ci in range(a.n_cells):
            model.layers[li][ci] = ComposableExpert(
                d=a.d, n_organelles=a.n, h=a.h,
                h_cell=a.h_cell, k=a.k,
                seq_len=a.seq_len, chunk_size=a.chunk,
                temp=a.temp, hinge_mode="loss",
                use_hinge_low=False, log_sharp=True,
                min_eff_organs=0,
                seed=a.seed * 1000 + li * 10 + ci,
                mem_budget_mb=a.mem_budget,
            ).to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n", type=int, default=384)          # 忆点总数
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--h", type=int, default=32)           # 共享基底宽度
    ap.add_argument("--h-cell", type=int, default=32)      # 每忆点隐层宽度
    ap.add_argument("--k", type=int, default=8)            # 拼几个
    ap.add_argument("--chunk", type=int, default=32)       # 显存安全阀
    ap.add_argument("--mem-budget", type=float, default=256.0,
                        help="拼装显存预算(MB)，超出自动切逐slot；"
                             "想让每token独立选专家(chunk=1)时它会自动生效")
    ap.add_argument("--n-cells", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--rank", type=int, default=4)         # 新版不用，留着兼容
    ap.add_argument("--temp", type=float, default=5.0)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--scaffold-top-k", type=int, default=32)
    ap.add_argument("--select-mode", default="soft", choices=["topk", "soft", "none"])
    ap.add_argument("--shared-mult", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default="output/v8")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--fix-mojibake", action="store_true")
    ap.add_argument("--no-leak-check", action="store_true")
    ap.add_argument("--save-ckpt", default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = (device == "cuda") and (not a.no_bf16)
    max_steps = a.max_steps
    tag = a.tag or ("v8_k%d_hc%d_s%d" % (a.k, a.h_cell, a.seed))

    print("=" * 70)
    print("  Snowflake v8 真·拼专家   tag=%s" % tag)
    print("  device=%s  bf16=%s  steps=%d" % (device, bf16, max_steps))
    print("  d=%d  n=%d  L=%d  h_base=%d  h_cell=%d  k=%d  chunk=%d"
          % (a.d, a.n, a.L, a.h, a.h_cell, a.k, a.chunk))
    print("  拼后等效隐层宽度 = k × h_cell + h_base = %d"
          % (a.k * a.h_cell + a.h))
    print("=" * 70, flush=True)

    get_batch, vocab_size = load_data(a.data, a.seq_len, a.bs, a.fix_mojibake)
    model = build_v8(a, vocab_size, device)
    params = sum(p.numel() for p in model.parameters())
    print("[model] 参数量 %s" % format(params, ","), flush=True)

    if vocab_size == 0:
        raise RuntimeError("vocab=0 —— 数据是空的，别往下跑")

    if not a.no_leak_check:
        print("\n[check] 因果泄漏自检 ...", flush=True)
        ok = leak_check(model, vocab_size=vocab_size,
                        seq_len=a.seq_len, B=2, device=device)
        if not ok:
            print("有泄漏，终止。")
            return

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=0.01, betas=(0.9, 0.95))
    warm = max(1, max_steps // 10)

    def lr_at(s):
        w = min(s / max(warm, 1), 1.0)
        return a.lr * (0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * w))))

    model.train()
    t0 = time.time()
    hist = []
    start_loss, final_loss = None, None

    for step in range(max_steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch("train")
        x, y = x.to(device), y.to(device)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits, diag = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                               y.view(-1))

        hs = model.heal_sum()
        if hs is not None:
            loss = loss + hs

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        lv = loss.item()
        if start_loss is None:
            start_loss = lv
        final_loss = lv

        if step % 10 == 0 or step == max_steps - 1:
            eff = diag.get("eff_organs")
            eff_s = "%.1f" % float(eff) if eff is not None else "-"
            vn = diag.get("wiring_var_norm")
            vn_s = "%.4f" % float(vn) if vn is not None else "-"
            gt = diag.get("gate")
            gt_s = "%.4f" % float(gt) if gt is not None else "-"
            print("  step %5d  loss=%.4f  lr=%.2e  eff=%s  var_norm=%s  gate=%s"
                  % (step, lv, lr, eff_s, vn_s, gt_s), flush=True)
            hist.append({"step": step, "loss": lv, "lr": lr,
                         "eff": float(eff) if eff is not None else None,
                         "var_norm": float(vn) if vn is not None else None,
                         "gate": float(gt) if gt is not None else None})

    elapsed = time.time() - t0
    val_loss = evaluate(model, get_batch, 20, device, bf16)

    def _last(key):
        for h in reversed(hist):
            if h.get(key) is not None:
                return h[key]
        return None

    eff_last = _last("eff")
    vn_last = _last("var_norm")
    gt_last = _last("gate")
    h2 = "pass" if (vn_last is not None and vn_last > 0.05) else \
         ("n/a" if vn_last is None else "fail")

    print("\n" + "=" * 70)
    print("  [%s] 完成" % tag)
    print("    起始 loss = %.4f" % start_loss)
    print("    末步 loss = %.4f" % final_loss)
    print("    val  loss = %.4f   ppl = %.4f" % (val_loss, math.exp(val_loss)))
    print("    eff_organs(末) = %s" % eff_last)
    print("    var_norm(末)   = %s   -> H2 %s  (阈值 0.05)" % (vn_last, h2))
    print("    gate(末)       = %s   <- 若仍≈0，说明模型没启用拼装" % gt_last)
    print("    参数量 = %s   用时 = %.1fs" % (format(params, ","), elapsed))
    print("    ln(vocab) = %.4f" % math.log(vocab_size))
    print("=" * 70)

    os.makedirs(a.outdir, exist_ok=True)
    rec = {
        "tag": tag, "config": vars(a), "vocab_size": vocab_size,
        "params": params, "start_loss": start_loss, "final_loss": final_loss,
        "val_loss": val_loss, "ppl": math.exp(val_loss),
        "eff_organs_last": eff_last, "var_norm_last": vn_last,
        "gate_last": gt_last, "H2": h2, "elapsed_sec": elapsed,
        "composed_width": a.k * a.h_cell + a.h,
        "history": hist,
    }
    out = os.path.join(a.outdir, "%s.json" % tag)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print("[saved] %s" % out)

    if a.save_ckpt:
        torch.save({"model": model.state_dict(), "config": vars(a)}, a.save_ckpt)
        print("[ckpt] %s" % a.save_ckpt)


if __name__ == "__main__":
    main()
