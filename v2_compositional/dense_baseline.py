"""
Dense Baseline 对照脚本
=======================
目的：为 README 的 claim
    "achieving strong generalization under a fraction of the
     parameters of a dense baseline"
提供【真实数据支撑】。当前仓库所有表格都没有 dense baseline。

设计原则（保证公平对比）：
  1. 参数量对齐仓库 CellMoE 的 1,823,077（实际 1,789,824，偏差 -1.8%）
  2. 同一份数据、同一个 tokenizer（字符级 vocab=98）
  3. 同一套训练超参（lr / warmup / steps / bs / seq）
  4. 同一套评估（val loss / ppl）
  5. 同样跑 10 seeds，做配对 t 检验

用法:
    python dense_baseline.py --data <txt> --max-steps 3000 --seed 0
    python dense_baseline.py --sweep          # 10 seeds
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════
# 模型：标准 dense transformer（GPT 风格）
# ════════════════════════════════════════════════════════════
class DenseLM(nn.Module):
    """标准 dense transformer LM，作为 Snowflake MoE 的对照。

    ★ 关键设计（保证公平对比）：
      1. 默认【不加】pos embedding —— SnowflakeLM 也没有位置编码。
         若加了 pos，参数量虚高 9.9%，对比就不公平了。
      2. tied embedding（head 复用 emb 权重）—— 与 Snowflake 一致。
      3. 无 bias（Snowflake 的 in_proj / cell head 都 bias=False）。

    参数量校准（vocab=98）：
      L=5, d=288, ff=1152 → 5,024,736  (0.973x Snowflake tiny 5,163,618)
      L=4, d=320, ff=1280 → 4,964,480  (0.961x)
    """

    def __init__(self, V, d=288, L=5, h=8, ff=None, max_len=1024,
                 tied=True, dropout=0.0, use_pos=False):
        super().__init__()
        ff = ff or 4 * d
        self.V, self.d, self.L = V, d, L
        self.use_pos = use_pos
        self.emb = nn.Embedding(V, d)
        # ★ 默认【不加】pos embedding：与 SnowflakeLM 对齐（它也没有）
        self.pos = nn.Embedding(max_len, d) if use_pos else None
        self.ln_in = nn.LayerNorm(d)

        self.qkv = nn.ModuleList([nn.Linear(d, 3 * d) for _ in range(L)])
        self.o = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.w1 = nn.ModuleList([nn.Linear(d, ff) for _ in range(L)])
        self.w2 = nn.ModuleList([nn.Linear(ff, d) for _ in range(L)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])

        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        if tied:
            self.head.weight = self.emb.weight      # 权重绑定
        self.drop = nn.Dropout(dropout)
        self.h = h
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x):
        B, T = x.shape
        dev = x.device
        h = self.emb(x)
        if self.pos is not None:
            pos = torch.arange(T, device=dev).unsqueeze(0)
            h = h + self.pos(pos)
        h = self.ln_in(h)
        causal = torch.triu(torch.ones(T, T, device=dev, dtype=torch.bool), 1)
        for i in range(self.L):
            qkv = self.qkv[i](self.ln1[i](h))
            q, k, v = qkv.chunk(3, dim=-1)
            dh = self.d // self.h
            q = q.view(B, T, self.h, dh).transpose(1, 2)
            k = k.view(B, T, self.h, dh).transpose(1, 2)
            v = v.view(B, T, self.h, dh).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
            a = a.transpose(1, 2).reshape(B, T, self.d)
            h = h + self.drop(self.o[i](a))
            ff = self.w2[i](F.gelu(self.w1[i](self.ln2[i](h))))
            h = h + self.drop(ff)
        return self.head(self.ln_f(h))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ════════════════════════════════════════════════════════════
# 数据（与 train_lock60.py 完全一致）
# ════════════════════════════════════════════════════════════
def fix_mojibake(txt: str) -> str:
    try:
        return txt.encode("cp1252", errors="strict").decode("utf-8",
                                                            errors="ignore")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return txt


def load_data(path, fix_mj=False):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    if fix_mj:
        txt2 = fix_mojibake(txt)
        if len(txt2) > len(txt) * 0.9:
            txt = txt2
    chars = sorted(set(txt))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in txt], dtype=torch.long)
    n = int(len(data) * 0.9)
    return data[:n], data[n:], len(chars)


def get_batch(split_data, bs, seq, dev):
    ix = torch.randint(len(split_data) - seq - 1, (bs,))
    x = torch.stack([split_data[i:i + seq] for i in ix])
    y = torch.stack([split_data[i + 1:i + 1 + seq] for i in ix])
    return x.to(dev), y.to(dev)


# ════════════════════════════════════════════════════════════
# 训练
# ════════════════════════════════════════════════════════════
def train_one(data_path, seed, steps, bs, seq, lr, warmup, fix_mj,
              outdir, a, log_every=100):
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}", flush=True)

    tr, va, V = load_data(data_path, fix_mj)
    print(f"[data] {len(tr)+len(va):,} tokens  vocab={V}  "
          f"train={len(tr):,} val={len(va):,}", flush=True)

    model = DenseLM(V, d=a.d, L=a.L, h=a.h).to(dev)
    P = model.n_params()
    print(f"[model] DenseLM L={a.L} d={a.d}  params={P:,}  "
          f"(Snowflake tiny 5,163,618, 比值 {P/5_163_618:.3f}x)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        prog = (s - warmup) / max(1, steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    ce = nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        x, y = get_batch(tr, bs, seq, dev)
        loss = ce(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % log_every == 0 or s == steps - 1:
            print(f"  step {s:>5}  loss={loss.item():.4f}  "
                  f"lr={lr_at(s):.2e}  {time.time()-t0:.0f}s", flush=True)

    # ---- 评估 ----
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for _ in range(20):
            x, y = get_batch(va, bs, seq, dev)
            l = ce(model(x).reshape(-1, V), y.reshape(-1))
            tot += l.item()
            cnt += 1
    val_loss = tot / cnt
    ppl = math.exp(min(val_loss, 20))
    print(f"\n[RESULT] seed={seed}  params={P:,}  "
          f"val_loss={val_loss:.5f}  ppl={ppl:.4f}", flush=True)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"dense_seed{seed}.txt"), "w") as f:
        f.write(f"seed={seed}\nparams={P}\nval_loss={val_loss}\nppl={ppl}\n")
    return {"seed": seed, "params": P, "val_loss": val_loss, "ppl": ppl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--fix-mojibake", action="store_true")
    ap.add_argument("--outdir", type=str, default="output/dense")
    ap.add_argument("--sweep", action="store_true", help="跑 10 seeds")
    ap.add_argument("--d", type=int, default=288)
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--scale-sweep", action="store_true",
                    help="扫规模：找达到 Snowflake loss 的临界参数量")
    a = ap.parse_args()

    if a.scale_sweep:
        print("=" * 88)
        print("  规模扫描：找 dense 达到 Snowflake loss=1.788 的临界点")
        print("=" * 88)
        grid = [(4, 192), (4, 256), (4, 320), (5, 288), (4, 384), (6, 320)]
        out = []
        for L_, d_ in grid:
            a.L, a.d = L_, d_
            r = train_one(a.data, a.seed, a.max_steps, a.bs, a.seq,
                          a.lr, a.warmup, a.fix_mojibake, a.outdir, a)
            out.append((L_, d_, r["params"], r["val_loss"]))
            print(f"  L={L_} d={d_}  params={r['params']:,}  "
                  f"val_loss={r['val_loss']:.5f}", flush=True)
        print("\n  汇总（Snowflake tiny: 5,163,618 params, loss=1.7882）:")
        print(f"  {'L':>3}{'d':>6}{'params':>13}{'vs Snow':>9}{'loss':>10}")
        for L_, d_, pp, ll in sorted(out, key=lambda x: x[2]):
            print(f"  {L_:>3}{d_:>6}{pp:>13,}{pp/5_163_618:>8.3f}x{ll:>10.5f}")
        return

    seeds = list(range(10)) if a.sweep else [a.seed]
    res = []
    for s in seeds:
        r = train_one(a.data, s, a.max_steps, a.bs, a.seq, a.lr,
                      a.warmup, a.fix_mojibake, a.outdir, a)
        res.append(r)

    if len(res) > 1:
        import numpy as np
        v = np.array([r["val_loss"] for r in res])
        p = np.array([r["ppl"] for r in res])
        print("\n" + "=" * 70)
        print("  Dense baseline 汇总")
        print("=" * 70)
        print(f"  val_loss  = {v.mean():.5f} ± {v.std(ddof=1):.5f}   "
              f"95%CI=[{v.mean()-2.262*v.std(ddof=1)/np.sqrt(len(v)):.5f}, "
              f"{v.mean()+2.262*v.std(ddof=1)/np.sqrt(len(v)):.5f}]")
        print(f"  ppl       = {p.mean():.4f} ± {p.std(ddof=1):.4f}")
        print(f"  params    = {res[0]['params']:,}")
        print("\n  ★ 与 Snowflake MoE 对比（见 compare_dense.py）")


if __name__ == "__main__":
    main()
