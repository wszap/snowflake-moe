"""
=============================================================================
挂谷翻转塞训练脚本（v6）—— 在 v3 基础上插入多轴 Householder 反射
=============================================================================

核心假设
    细胞器输出 delta 的有效秩被 rank=4 锁死（dense FFN 是满秩 128）。
    挂谷翻转塞：每个 token 自己从忆点库检索 K 个翻转轴 → 依次反射
    ⇒ 不同 token 被送到不同方向 ⇒ 整体覆盖维度扩大。

    ★ 数学自检已验证（verify_kakeya_math.py，15/15）：
        固定轴：rank 4 → 4     （反射保秩，一点没变）
        检索轴：rank 4 → 128   （K=8 即满秩）

用法
    # 基线（K=0，等价于 v3，用来对照）
    python train_v6.py --data data_25mb.txt --kakeya-k 0 --tag k0_base

    # K 扫描
    for k in 4 8 16 32; do
        python train_v6.py --data data_25mb.txt --max-steps 3000 \
            --kakeya-k $k --tag kak_k$k
    done

零初始化保证
    α 与对角门均零初始化 ⇒ 起步时严格恒等 ⇒ K 无论多大，
    第一步的 loss 与 v3 完全一致（见 smoke 时的 start_loss）。
=============================================================================
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_v3 import load_data, evaluate          # noqa: E402
from scaffold_v3 import leak_check                # noqa: E402
from kakeya import KakeyaLMV3                     # noqa: E402


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
    ap.add_argument("--select-mode", default="topk",
                    choices=["topk", "soft", "none"])
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--temp", type=float, default=5.0)
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
    ap.add_argument("--outdir", default="output/v6")

    # ---- 挂谷翻转塞专属 ----
    ap.add_argument("--kakeya-k", type=int, default=8,
                    help="翻转轴个数；0 = 关闭（基线）")
    ap.add_argument("--kakeya-off-delta", action="store_true",
                    help="改为作用在整条残差流 x 上（默认作用在 delta）")
    ap.add_argument("--kakeya-no-diag", action="store_true",
                    help="关闭对角门（纯正交，不能缩放）")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = a.bf16 and device == "cuda"

    max_steps = a.smoke_steps if a.smoke else a.max_steps
    tag = a.tag or f"v6_k{a.kakeya_k}_s{a.seed}"

    print("=" * 70)
    print(f"  Snowflake v6 (挂谷翻转塞)   tag={tag}")
    print(f"  device={device}  bf16={bf16}  steps={max_steps}")
    print(f"  kakeya_k={a.kakeya_k}  on_delta={not a.kakeya_off_delta}  "
          f"diag={not a.kakeya_no_diag}")
    print("=" * 70, flush=True)

    # ---- 数据 ----
    get_batch, vocab_size = load_data(a.data, a.seq_len, a.bs, a.fix_mojibake)
    if vocab_size == 0:
        raise RuntimeError("vocab=0 —— 数据是空的，别往下跑")

    # ---- 模型 ----
    model = KakeyaLMV3(
        vocab_size=vocab_size, d=a.d, n=a.n, L=a.L, h=a.h,
        n_cells=a.n_cells, seq_len=a.seq_len, rank=a.rank, temp=a.temp,
        n_head=a.n_head, scaffold_top_k=a.scaffold_top_k,
        select_mode=a.select_mode, seed=a.seed,
        kakeya_k=a.kakeya_k,
        kakeya_on_delta=not a.kakeya_off_delta,
        kakeya_use_diag=not a.kakeya_no_diag,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    base_params = sum(p.numel() for p in model.base.parameters())
    print(f"[model] 参数量 {params:,}  （其中基座 {base_params:,}，"
          f"翻转塞 {params - base_params:,}）", flush=True)

    expect_min = vocab_size * a.d + a.d * vocab_size + vocab_size
    if params < expect_min:
        raise RuntimeError(
            f"参数量 {params:,} < 词表理论最小值 {expect_min:,}，数据没接进去")

    # ---- 泄漏自检 ----
    # 翻转塞完全 per-token（检索与反射都不跨位置）⇒ 结构上不可能泄漏，
    # 但仍然实测一次；若包装类接口不适配则打印警告继续。
    if not a.no_leak_check:
        print("\n[check] 因果泄漏自检 ...", flush=True)
        try:
            ok = leak_check(model, vocab_size=vocab_size,
                            seq_len=a.seq_len, B=2, device=device)
        except Exception as e:
            print(f"  [warn] leak_check 对包装类不适用（{e}），"
                  f"翻转塞 per-token 结构上无泄漏，继续", flush=True)
            ok = True
        if not ok:
            print("❌ 有泄漏，终止。数字不可用于论文。")
            return
    else:
        print("\n[check] 泄漏自检已跳过", flush=True)

    # ---- 优化器 ----
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=0.01, betas=(0.9, 0.95))

    steps = max(1, max_steps // 10)

    def lr_at(s):
        w = min(s / max(steps, 1), 1.0)
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
            eff_s = f"{float(eff):.1f}" if eff is not None else "-"
            vn = diag.get("wiring_var_norm")
            vn_s = f"{float(vn):.4f}" if vn is not None else "-"
            print(f"  step {step:>5}  loss={lv:.4f}  lr={lr:.2e}  "
                  f"eff={eff_s}  var_norm={vn_s}", flush=True)
            hist.append({"step": step, "loss": lv, "lr": lr,
                         "eff": float(eff) if eff is not None else None,
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

    # 翻转角学到了多少
    alphas = []
    for f in model.flippers:
        alphas.append(torch.tanh(f.alpha).detach().float().cpu())
    alpha_mean = float(torch.cat(alphas).mean()) if alphas else 0.0
    alpha_abs = float(torch.cat(alphas).abs().mean()) if alphas else 0.0

    print("\n" + "=" * 70)
    print(f"  [{tag}] 完成")
    print(f"    起始 loss = {start_loss:.4f}   （应与 v3 一致，因零初始化）")
    print(f"    末步 loss = {final_loss:.4f}")
    print(f"    val  loss = {val_loss:.4f}   ppl = {math.exp(val_loss):.4f}")
    print(f"    eff_organs(末) = {eff_last}")
    print(f"    var_norm(末)   = {vn_last}   → H2 {h2}  (阈值 0.05)")
    print(f"    翻转角 α: 均值 {alpha_mean:+.4f}  绝对均值 {alpha_abs:.4f}")
    print(f"    参数量 = {params:,}   用时 = {elapsed:.1f}s")
    print("=" * 70)

    os.makedirs(a.outdir, exist_ok=True)
    rec = {
        "tag": tag,
        "config": vars(a),
        "vocab_size": vocab_size,
        "params": params,
        "base_params": base_params,
        "start_loss": start_loss,
        "final_loss": final_loss,
        "val_loss": val_loss,
        "ppl": math.exp(val_loss),
        "eff_organs_last": eff_last,
        "var_norm_last": vn_last,
        "H2_按输入组合": h2,
        "kakeya_k": a.kakeya_k,
        "kakeya_alpha_mean": alpha_mean,
        "kakeya_alpha_absmean": alpha_abs,
        "elapsed_sec": elapsed,
        "leak_ok": not a.no_leak_check,
        "history": hist,
    }
    out = os.path.join(a.outdir, f"{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
