# -*- coding: utf-8 -*-
"""v4 训练脚本。与 train_v3.py 完全一致，只是换成 SnowflakeLMV4。

复用 train_v3 的 load_data / evaluate，保证数据管线与 v3 完全一致。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--scaffold-top-k", type=int, default=32)
    ap.add_argument("--select-mode", default="soft",
                    choices=["topk", "soft", "none"])
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--temp", type=float, default=5.0)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--shared-mult", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-bf16", dest="bf16", action="store_false",
                    default=True)
    ap.add_argument("--fix-mojibake", action="store_true")
    ap.add_argument("--no-leak-check", action="store_true")
    ap.add_argument("--save-ckpt", default="")
    ap.add_argument("--outdir", default="output/v4")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = a.bf16 and device == "cuda"

    max_steps = a.smoke_steps if a.smoke else a.max_steps
    tag = a.tag or ("v4_%s_s%d" % (a.variant.replace(",", "+"), a.seed))

    print("=" * 70)
    print("  Snowflake v4   tag=%s" % tag)
    print("  device=%s  bf16=%s  steps=%d" % (device, bf16, max_steps))
    print("  variant=%s  shared_mult=%d  rank=%d  temp=%s  d=%d"
          % (a.variant, a.shared_mult, a.rank, a.temp, a.d))
    print("=" * 70, flush=True)

    get_batch, vocab_size = load_data(a.data, a.seq_len, a.bs, a.fix_mojibake)

    model = SnowflakeLMV4(
        vocab_size=vocab_size, d=a.d, n=a.n, L=a.L, h=a.h,
        n_cells=a.n_cells, seq_len=a.seq_len, rank=a.rank, temp=a.temp,
        n_head=a.n_head, scaffold_top_k=a.scaffold_top_k,
        select_mode=a.select_mode, variant=a.variant,
        shared_mult=a.shared_mult, seed=a.seed,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print("[model] 参数量 %s" % format(params, ","), flush=True)

    if vocab_size == 0:
        raise RuntimeError("vocab=0 —— 数据是空的，别往下跑")
    expect_min = vocab_size * a.d + a.d * vocab_size + vocab_size
    if params < expect_min:
        raise RuntimeError(
            "参数量 %s < 词表理论最小值 %s，数据很可能没接进去"
            % (format(params, ","), format(expect_min, ",")))

    if not a.no_leak_check:
        print("\n[check] 因果泄漏自检 ...", flush=True)
        ok = leak_check(model, vocab_size=vocab_size,
                        seq_len=a.seq_len, B=2, device=device)
        if not ok:
            print("有泄漏，终止。")
            return
    else:
        print("\n[check] 泄漏自检已跳过", flush=True)

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
            sh = diag.get("mix_sharp")
            sh_s = "%.3f" % float(sh) if sh is not None else "-"
            vn = diag.get("wiring_var_norm")
            vn_s = "%.4f" % float(vn) if vn is not None else "-"
            print("  step %5d  loss=%.4f  lr=%.2e  eff=%s  sharp=%s  "
                  "var_norm=%s" % (step, lv, lr, eff_s, sh_s, vn_s),
                  flush=True)
            hist.append({"step": step, "loss": lv, "lr": lr,
                         "eff": float(eff) if eff is not None else None,
                         "mix_sharp": float(sh) if sh is not None else None,
                         "var_norm": float(vn) if vn is not None else None})

    elapsed = time.time() - t0
    val_loss = evaluate(model, get_batch, 20, device, bf16)

    def _last(key):
        for h in reversed(hist):
            if h.get(key) is not None:
                return h[key]
        return None

    eff_last = _last("eff")
    vn_last = _last("var_norm")
    h2 = "pass" if (vn_last is not None and vn_last > 0.05) else \
         ("n/a" if vn_last is None else "fail")

    print("\n" + "=" * 70)
    print("  [%s] 完成" % tag)
    print("    起始 loss = %.4f" % start_loss)
    print("    末步 loss = %.4f" % final_loss)
    print("    val  loss = %.4f   ppl = %.4f" % (val_loss, math.exp(val_loss)))
    print("    eff_organs(末) = %s" % eff_last)
    print("    var_norm(末)   = %s   -> H2 %s  (阈值 0.05)" % (vn_last, h2))
    print("    参数量 = %s   用时 = %.1fs" % (format(params, ","), elapsed))
    print("    ln(vocab) = %.4f" % math.log(vocab_size))
    print("=" * 70)

    os.makedirs(a.outdir, exist_ok=True)
    rec = {
        "tag": tag,
        "config": vars(a),
        "vocab_size": vocab_size,
        "params": params,
        "start_loss": start_loss,
        "final_loss": final_loss,
        "val_loss": val_loss,
        "ppl": math.exp(val_loss),
        "eff_organs_last": eff_last,
        "var_norm_last": vn_last,
        "H2": h2,
        "elapsed_sec": elapsed,
        "leak_ok": not a.no_leak_check,
        "history": hist,
    }
    out = os.path.join(a.outdir, "%s.json" % tag)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print("[saved] %s" % out)

    if a.save_ckpt:
        torch.save({"model": model.state_dict(), "config": vars(a)},
                   a.save_ckpt)
        print("[ckpt] %s" % a.save_ckpt)


if __name__ == "__main__":
    main()
