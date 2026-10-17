"""
=============================================================================
Snowflake 架构 v3 训练脚本 —— 核架(关联+筛选) → 忆点 → 细胞器 → 安全机制
=============================================================================

用法
    # smoke（2 分钟，先验证链路）
    python train_v3.py --data data_25mb.txt --smoke --L 2 --n-cells 2

    # 正式跑
    python train_v3.py --data data_25mb.txt --max-steps 3000 --seed 0

    # 核心消融：筛选门到底值多少
    for m in topk soft none; do
        python train_v3.py --data data_25mb.txt --max-steps 3000 \
            --select-mode $m --tag sel_$m
    done

=============================================================================
与 v2 (train_lock60.py) 的关键差别
=============================================================================
    + 核架 Scaffold：全文因果关联 → 先筛选出最相关的 → 再给忆点
    + 位置编码
    + 残差连接
    - chunk 平均（泄漏源）：chunk_size=1 后自动消失

=============================================================================
两道防线（踩过 5 次空数据坑，这次写死在代码里）
=============================================================================
    ① 数据加载后立刻 assert 字节数 —— 空文件直接崩，不给你跑 3000 步
    ② 参数量指纹校验 —— vocab=0 时参数量会少 25,186，一眼可辨
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

from scaffold_v3 import SnowflakeLMV3, leak_check   # noqa: E402


# ============================================================================
# 数据加载（含空数据防线 ①）
# ============================================================================
def load_data(path, seq_len, bs, fix_mojibake=False):
    """字符级加载。★ 写完立刻 assert，源文件缺失/为空时直接崩。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")

    raw = open(path, "rb").read()
    txt = raw.decode("utf-8", errors="ignore")

    if fix_mojibake:
        try:
            cand = txt.encode("latin-1").decode("utf-8")
            if len(set(cand)) < len(set(txt)):
                txt = cand
        except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
            pass

    n_bytes = len(raw)
    # ★ 防线 ①：空文件/过小文件直接崩，不让训练白跑
    assert n_bytes >= 1_000_000, \
        f"[数据为空或过小] {path} 仅 {n_bytes} 字节。请用魔搭重新生成数据。"

    chars = sorted(set(txt))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in txt], dtype=torch.long)
    V = len(chars)

    n = int(len(ids) * 0.9)
    tr, va = ids[:n], ids[n:]

    print(f"[data] {len(ids):,} tokens  vocab={V}  "
          f"train={len(tr):,} val={len(va):,}  ({n_bytes:,} bytes)",
          flush=True)

    def get_batch(split):
        src = tr if split == "train" else va
        ix = torch.randint(len(src) - seq_len - 1, (bs,))
        x = torch.stack([src[i:i + seq_len] for i in ix])
        y = torch.stack([src[i + 1:i + 1 + seq_len] for i in ix])
        return x, y

    return get_batch, V


@torch.no_grad()
def evaluate(model, get_batch, n_batch, device, bf16):
    model.eval()
    tot, cnt = 0.0, 0
    for _ in range(n_batch):
        x, y = get_batch("val")
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                               y.view(-1))
        tot += loss.item(); cnt += 1
    model.train()
    return tot / max(cnt, 1)


# ============================================================================
# 主流程
# ============================================================================
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
    ap.add_argument("--save-ckpt", default="")
    ap.add_argument("--outdir", default="output/v3")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = a.bf16 and device == "cuda"

    max_steps = a.smoke_steps if a.smoke else a.max_steps
    tag = a.tag or (f"v3_{a.select_mode}_s{a.seed}")

    print("=" * 70)
    print(f"  Snowflake v3   tag={tag}")
    print(f"  device={device}  bf16={bf16}  steps={max_steps}")
    print(f"  select_mode={a.select_mode}  scaffold_top_k={a.scaffold_top_k}")
    print("=" * 70, flush=True)

    # ---- 数据 ----
    get_batch, vocab_size = load_data(a.data, a.seq_len, a.bs, a.fix_mojibake)

    # ---- 模型 ----
    model = SnowflakeLMV3(
        vocab_size=vocab_size, d=a.d, n=a.n, L=a.L, h=a.h,
        n_cells=a.n_cells, seq_len=a.seq_len, rank=a.rank, temp=a.temp,
        n_head=a.n_head, scaffold_top_k=a.scaffold_top_k,
        select_mode=a.select_mode, seed=a.seed,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"[model] 参数量 {params:,}", flush=True)

    # ★ 防线 ②：参数量指纹。vocab=0 时 embed+head 会少约 25,186
    #   （本次测试模型 d=128, vocab=100 → embed 12,800 + head 12,900）
    if vocab_size == 0:
        raise RuntimeError("vocab=0 —— 数据是空的，别往下跑")
    expect_min = vocab_size * a.d + a.d * vocab_size + vocab_size
    if params < expect_min:
        raise RuntimeError(
            f"参数量 {params:,} < 词表部分理论最小值 {expect_min:,}，"
            f"数据很可能没接进去")

    # ---- 泄漏自检（补完核架后必做）----
    if not a.no_leak_check:
        print("\n[check] 因果泄漏自检 ...", flush=True)
        ok = leak_check(model, vocab_size=vocab_size,
                        seq_len=a.seq_len, B=2, device=device)
        if not ok:
            print("❌ 有泄漏，终止。数字不可用于论文。")
            return
    else:
        print("\n[check] 泄漏自检已跳过（--no-leak-check）", flush=True)

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

        # 细胞器的 heal 项（若有）
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
            sh = diag.get("mix_sharp")
            sh_s = f"{float(sh):.3f}" if sh is not None else "-"
            vn = diag.get("wiring_var_norm")
            vn_s = f"{float(vn):.4f}" if vn is not None else "-"
            print(f"  step {step:>5}  loss={lv:.4f}  lr={lr:.2e}  "
                  f"eff={eff_s}  sharp={sh_s}  var_norm={vn_s}", flush=True)
            hist.append({"step": step, "loss": lv, "lr": lr,
                         "eff": float(eff) if eff is not None else None,
                         "mix_sharp": float(sh) if sh is not None else None,
                         "var_norm": float(vn) if vn is not None else None})

    elapsed = time.time() - t0

    # ---- 收尾 ----
    val_loss = evaluate(model, get_batch, 20, device, bf16)

    def _last(key):
        for h in reversed(hist):
            if h.get(key) is not None:
                return h[key]
        return None

    eff_last = _last("eff")
    vn_last = _last("var_norm")

    # H2 判据：wiring_var_norm > 0.05 才算「按输入组合」
    h2 = "pass" if (vn_last is not None and vn_last > 0.05) else \
         ("n/a" if vn_last is None else "fail")

    print("\n" + "=" * 70)
    print(f"  [{tag}] 完成")
    print(f"    起始 loss = {start_loss:.4f}")
    print(f"    末步 loss = {final_loss:.4f}")
    print(f"    val  loss = {val_loss:.4f}   ppl = {math.exp(val_loss):.4f}")
    print(f"    eff_organs(末) = {eff_last}")
    print(f"    var_norm(末)   = {vn_last}   → H2 {h2}  (阈值 0.05)")
    print(f"    参数量 = {params:,}   用时 = {elapsed:.1f}s")
    print(f"    ln(vocab) = {math.log(vocab_size):.4f}  "
          f"(起始 loss 应接近它)")
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
        "H2_按输入组合": h2,
        "elapsed_sec": elapsed,
        "leak_ok": not a.no_leak_check,
        "history": hist,
    }
    out = os.path.join(a.outdir, f"{tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out}")

    if a.save_ckpt:
        torch.save({"model": model.state_dict(), "config": vars(a)}, a.save_ckpt)
        print(f"[ckpt] {a.save_ckpt}")


if __name__ == "__main__":
    main()
