# -*- coding: utf-8 -*-
"""
ablate_permute.py —— 分工真实性消融（真机跑，需 torch + GPU）

=============================================================================
这一跑回答的唯一问题：mix 到底有没有携带样本相关信息？
=============================================================================

三个变体（同一份训练好的权重，只改 mix 的来源）：

    learned  : mix = softmax(x_chunk @ key.T / sqrt(d) * temp)      ← 正常
    constant : mix = mix_ema（所有 chunk 共用一份平均配方）          ← 完全退化
    permuted : mix = learned 的行随机重排                            ← 破坏内容↔配方对应

判据：
    mix_gain(learned)  = loss(constant) − loss(learned)
    mix_gain(permuted) = loss(constant) − loss(permuted)
    核心量 = mix_gain(learned) − mix_gain(permuted) = loss(permuted) − loss(learned)

    核心量 > 0 且 CI 下界 > 0  ⇒  "分工"主张成立
    核心量 ≈ 0                ⇒  只能说"未检出分工"，不能说"证明没有分工"

=============================================================================
⚠ 不要做「训练时打乱器官顺序」这个变体
=============================================================================
已验证（permute_check.py）：softmax 与 einsum 对器官维都是置换等变的，
一致地 permute (organelle_memory, W1, W2) 后输出【逐位不变】（差 2.5e-16）。
按字面实现，loss 曲线会与主组完全相同 —— 白跑，且会让你误以为"分工不存在"。

=============================================================================
本脚本同时修复 lock47 → lock48 的三处（否则 N=384 训练会被 heal 主导）
=============================================================================
1. heal 归一化 /N → /[N(N-1)]
   原版 off-diagonal 有 N(N-1) 项却只除以 N ⇒ heal 随 N 线性爆炸：
   N=8 → 0.58（无害）；N=384 → 37.5（是 ce_loss 2.3 的 16 倍，完全主导训练）
2. organelle_memory 改用共享基初始化（不是 randn*0.5）
   M = b_norm · sqrt(band/(1−band)) · s · sqrt(d) + noise·s
   实测：随机初始化 in_band≈0.0004，共享基 → in_band≈0.999（与 N 无关）
3. 低秩 delta 的量级必须与 base 同量级，否则加权和被 base 主导、
   合成权重永远谱平坦（条件数≈1，判据失效）

用法：
    python ablate_permute.py --n 384 --p 32 --seeds 10
    python ablate_permute.py --n 384 --p 128          # seq_len 级 mix 对比
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_logger import RunLogger, STEP_FIELDS, FINAL_FIELDS  # noqa: E402

# 把 lock48 新增字段注册进 schema（不注册会被 logger 拒绝）
STEP_FIELDS.update({"mix_gain": "float", "core_stat": "float"})
FINAL_FIELDS.update({"core_stat": "float", "core_stat_ci_lo": "float",
                     "core_stat_ci_hi": "float", "cohens_dz": "float",
                     "p_core": "float", "n_permute_repeats": "int"})


# --------------------------------------------------------------------------
# 帐篷 + 铰链（lock48 修正版）
# --------------------------------------------------------------------------
def tent_hinge(mem, band=0.5, width=0.1, rho=0.3):
    """返回 (u, c, heal_term, in_band)。

    与 lock47 的两处差别：
      · heal 除以 N(N-1) 而非 N
      · 对角线 mask（对角 u_ii 是常数、梯度恒 0，mask 只影响显示值）
    """
    n = mem.shape[0]
    m = F.normalize(mem, dim=-1)
    G = m @ m.T
    u = (G - band) / width

    c = F.relu(2.0 - u.abs())
    c = c / (c.sum(-1, keepdim=True) + rho) * rho

    off = ~torch.eye(n, dtype=torch.bool, device=mem.device)
    ex = (F.relu(u.abs() - 2.0) ** 2) * off.float()
    heal = ex.sum() * (width ** 2) / (n * (n - 1))       # ← 修法：O(1)，与 N 无关

    in_band = (u.abs() < 2.0).float()[off].mean()
    return u, c, heal, in_band


def shared_base_init(n, d, band=0.5, noise=0.5, seed=0):
    """共享基初始化：使两两 cos 起点就在带中心。

    cos ≈ a²/(a² + s²d)，取 a = sqrt(band/(1-band))·s·sqrt(d)
    """
    g = torch.Generator().manual_seed(seed)
    s = noise
    a = math.sqrt(band / (1.0 - band)) * s * math.sqrt(d)
    b = torch.randn(d, generator=g)
    b = b / b.norm()
    return b * a + torch.randn(n, d, generator=g) * s


def effective_rank(M, eps=1e-12):
    s = torch.linalg.svdvals(M)
    p = s / s.sum().clamp_min(eps)
    return torch.exp(-(p * (p + eps).log()).sum())


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
class PZExpert(nn.Module):
    def __init__(self, d, n=384, rank=4, chunk=32, seq_len=128, h=32,
                 band=0.5, width=0.1, rho=0.3, temp=2.0,
                 delta_ratio=1.0, init_seed=0):
        super().__init__()
        self.d, self.n, self.rank, self.h = d, n, rank, h
        self.seq_len = seq_len
        self.chunk = chunk if (chunk > 1 and seq_len % chunk == 0) else 1
        self.band, self.width, self.rho, self.temp = band, width, rho, temp

        # 器官签名（tent 的作用对象）——共享基初始化
        self.organelle_memory = nn.Parameter(
            shared_base_init(n, d, band=band, noise=0.5, seed=init_seed))

        # base + 低秩 delta。delta_ratio=1.0 ⇒ delta 与 base 同量级
        sc = 2.0 / math.sqrt(d)
        self.W1_base = nn.Parameter(torch.randn(d, h) * sc)
        self.W1_U = nn.Parameter(torch.randn(n, d, rank) * sc * delta_ratio)
        self.W1_V = nn.Parameter(torch.randn(rank, h) / math.sqrt(rank))
        sc2 = 2.0 / math.sqrt(h)
        self.W2_base = nn.Parameter(torch.randn(h, d) * sc2)
        self.W2_U = nn.Parameter(torch.randn(n, h, rank) * sc2 * delta_ratio)
        self.W2_V = nn.Parameter(torch.randn(rank, d) / math.sqrt(rank))

        self.head = nn.Linear(d, d, bias=False)
        self.register_buffer("mix_ema", torch.ones(n) / n)
        self.heal_term = None
        self._last_mix = None
        self._diag = {}

    @property
    def W1_all(self):
        return self.W1_base.unsqueeze(0) + (self.W1_U @ self.W1_V)

    @property
    def W2_all(self):
        return self.W2_base.unsqueeze(0) + (self.W2_U @ self.W2_V)

    def _chunk_view(self, x):
        N, d = x.shape
        if self.chunk <= 1 or N % self.seq_len != 0:
            return x.unsqueeze(1)
        B = N // self.seq_len
        return x.view(B, self.seq_len // self.chunk, self.chunk, d).reshape(
            -1, self.chunk, d)

    def compute_mix(self, x, perm=None, force_mix=None):
        """[N,d] -> [C,n]。perm/force_mix 只在评估时用。"""
        u, c, heal, in_band = tent_hinge(self.organelle_memory,
                                         self.band, self.width, self.rho)
        self.heal_term = heal
        key = self.organelle_memory + c.T @ self.organelle_memory   # 务必 c.T

        xc = self._chunk_view(x)
        C = xc.shape[0]
        xsum = xc.mean(1)
        mix = F.softmax(xsum @ key.T / math.sqrt(self.d) * self.temp, dim=-1)
        if force_mix is not None:
            fm = force_mix.unsqueeze(0) if force_mix.dim() == 1 else force_mix
            mix = fm.to(mix.dtype).expand(C, -1)
        elif perm is not None:
            mix = mix[perm]                    # ← 唯一有效的 permuted 对照

        # 更新 EMA 配方（constant 变体的参照物）。
        # 不更新的话 mix_ema 永远是初始均匀分布 1/n，
        # 那 constant 变体对比的是"均匀"而非"训练出的平均配方" —— 对照不干净。
        with torch.no_grad():
            bm = mix.detach().mean(0).to(self.mix_ema.dtype)
            self.mix_ema.mul_(0.99).add_(bm * 0.01)
        return mix, {"in_band": in_band, "heal_term": heal}

    def forward(self, x, perm=None, force_mix=None):
        N, d = x.shape
        mix, dg = self.compute_mix(x, perm=perm, force_mix=force_mix)
        xc = self._chunk_view(x)
        C, P, _ = xc.shape

        W1_all, W2_all = self.W1_all, self.W2_all
        W1w = torch.einsum("ci,idh->cdh", mix, W1_all)
        hh = torch.bmm(xc, W1w)
        hh = F.silu(hh)
        W2w = torch.einsum("ci,ihd->chd", mix, W2_all)
        out = torch.bmm(hh, W2w).reshape(N, d)

        md = mix.detach()
        self._last_mix = md
        self._diag = {
            "in_band": dg["in_band"].detach(), "heal_term": dg["heal_term"].detach(),
            "mix_var": md.var(0).mean().detach(),
            "mix_sharp": md.max(-1).values.mean().detach(),
            "mix_eff_rank": effective_rank(md[:512]).detach(),
        }
        return self.head(out), self._diag


# --------------------------------------------------------------------------
# 三变体评估
# --------------------------------------------------------------------------
@torch.no_grad()
def eval_variants(model, data_iter, loss_fn, n_batch=50, n_perm=20, seed=0):
    """返回 (loss_learned, loss_constant, loss_permuted_mean, loss_permuted_sd)"""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    L = {"learned": [], "constant": [], "permuted": []}

    for bi, (x, y) in enumerate(data_iter):
        if bi >= n_batch:
            break
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1)

        o, _ = model(x)
        L["learned"].append(loss_fn(o, y).item())

        o, _ = model(x, force_mix=model.mix_ema)
        L["constant"].append(loss_fn(o, y).item())

        for _ in range(n_perm):
            # 只在 batch 内重排（保持 mix 边缘分布完全不变）
            C = x.shape[0] // max(1, model.chunk)
            perm = torch.randperm(C, generator=g)
            o, _ = model(x, perm=perm)
            L["permuted"].append(loss_fn(o, y).item())

    out = {k: float(np.mean(v)) for k, v in L.items()}
    out["permuted_sd"] = float(np.std(L["permuted"]))
    return out


# --------------------------------------------------------------------------
# smoke：用【已知有分工】的合成数据跑完整闭环，验证判据能检出
# 构造：K 个簇，每簇一个真专家 ⇒ 最优 mix 必须按簇指派
# --------------------------------------------------------------------------
class ClusterTask:
    def __init__(self, n_cls=8, d=64, h=32, sep=2.0, n=4096, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.n_cls, self.d, self.h = n_cls, d, h
        self.centers = torch.randn(n_cls, d, generator=g) * sep
        self.Wstar = torch.randn(n_cls, d, h, generator=g) * 0.2
        self.Wout = torch.randn(h, d, generator=g) / math.sqrt(h)
        c = torch.randint(0, n_cls, (n,), generator=g)
        x = self.centers[c] + torch.randn(n, d, generator=g) * 0.5
        hh = torch.tanh(torch.einsum("bd,bdh->bh", x, self.Wstar[c]))
        self.x, self.y, self.c = x, hh @ self.Wout, c

    def batches(self, bs=256, shuffle=True):
        n = self.x.shape[0]
        idx = torch.randperm(n) if shuffle else torch.arange(n)
        for i in range(0, n, bs):
            j = idx[i:i + bs]
            yield self.x[j], self.y[j], self.c[j]


def smoke(n=8, d=64, h=32, steps=300, lr=0.01, n_perm=20, sep=2.0):
    """端到端：训练 → 三变体评估 → 判据。已知有分工，判据必须喊'有'。"""
    torch.manual_seed(0)
    task = ClusterTask(n_cls=n, d=d, h=h, sep=sep)
    model = PZExpert(d, n=n, rank=4, chunk=1, seq_len=1, h=h, temp=4.0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"\n[smoke] 合成任务：{n} 簇 / 每簇一个真专家 / sep={sep}（已知有分工）")
    for s in range(steps):
        tot = 0.0
        for xb, yb, _ in task.batches():
            opt.zero_grad()
            o, dg = model(xb)
            loss = loss_fn(o, yb) + 1.0 * model.heal_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        if s % 100 == 0 or s == steps - 1:
            print(f"  step {s:>4}  loss={tot:.5f}  "
                  f"in_band={dg['in_band']:.3f}  heal={dg['heal_term']:.5f}  "
                  f"sharp={dg['mix_sharp']:.3f}")

    # 三变体
    @torch.no_grad()
    def run(mode):
        L = []
        for xb, yb, _ in task.batches(bs=512, shuffle=False):
            if mode == "learned":
                o, _ = model(xb)
            elif mode == "constant":
                o, _ = model(xb, force_mix=model.mix_ema)
            else:
                # 注意：perm 长度必须是 chunk 数 C，不是样本数 N
                C = xb.shape[0] // max(1, model.chunk)
                o, _ = model(xb, perm=torch.randperm(C))
            L.append(loss_fn(o, yb).item())
        return float(np.mean(L))

    l_l = run("learned")
    l_c = run("constant")
    l_p = float(np.mean([run("permuted") for _ in range(n_perm)]))

    print(f"\n{'变体':<28}{'MSE':>14}{'mix_gain':>14}")
    print("  " + "-" * 54)
    print(f"  {'learned（学出配方）':<26}{l_l:>14.6f}{l_c - l_l:>14.6f}")
    print(f"  {'permuted（破坏配对）':<26}{l_p:>14.6f}{l_c - l_p:>14.6f}")
    print(f"  {'constant（完全退化）':<26}{l_c:>14.6f}{0.0:>14.6f}")

    core = l_p - l_l      # = mix_gain(learned) − mix_gain(permuted)
    print(f"\n  核心判据 loss(permuted) − loss(learned) = {core:+.6f}")
    verdict = "✅ 检出分工（判据有效）" if core > 1e-4 else \
              "⚠ 未检出 —— 但先确认构造真的有分工，别急着说'没有分工'"
    print(f"  判定：{verdict}")
    print("\n  → 这个 smoke 是【判据的阳性对照】。真机上若核心量 ≈ 0，")
    print("    说明模型没学到分工，而非判据坏了（判据已在阳性对照上验证过）。")
    return core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--p", type=int, default=32)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=10,
                    help="功率分析：n=5 只有 0.61 power，建议 >=10")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--root", default="output/runs")
    ap.add_argument("--smoke", action="store_true",
                    help="合成数据端到端自检（判据阳性对照），先跑这个")
    a = ap.parse_args()

    if a.smoke:
        smoke(n=8, d=64, h=32)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cfg] N={a.n} P={a.p} rank={a.rank} seeds={a.seeds} device={device}")

    # FLOPs 账（先看这个，别跑完才发现更贵）
    base_topk = 4 * a.d * a.h
    pz = (a.n * a.d * a.rank + a.d * a.rank * a.h) / a.p + a.d * a.h
    print(f"[FLOPs] 拼专家/token={pz:,.0f}  Top-K k=4/token={base_topk:,.0f}  "
          f"ratio={pz/base_topk:.3f}x")
    if pz > base_topk:
        print("  ⚠ 比 Top-K 更贵 —— 检查 rank 与 P。低秩是这个配置成立的前提。")

    results = []
    for seed in range(a.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = PZExpert(a.d, n=a.n, rank=a.rank, chunk=a.p,
                         seq_len=128, h=a.h, init_seed=seed).to(device)

        # ---- TODO: 接入你的真实训练循环 ----
        # 这里只搭好骨架；把 train_one() 里的 model/batch/loss 换进来即可。
        # 训练时必须：loss = ce + LAMBDA_HEAL * model.heal_term
        # 且每个 opt.step() 后调用 refresh_param_diag（见 pz_expert_ultimate.py）
        raise SystemExit(
            "[STOP] 尚未接入真实训练循环。\n"
            "  请把 lock47 的 train_one() 搬进来，改动三处：\n"
            "    1) model = PZExpert(...)  （本文件，已含 3 处 lock48 修正）\n"
            "    2) loss = ce + LAMBDA_HEAL * model.heal_term   （heal 已 O(1)）\n"
            "    3) 训练结束后调用 eval_variants(model, ...) 拿三变体 loss\n"
            "  删掉本 SystemExit 即可运行。")


if __name__ == "__main__":
    main()
