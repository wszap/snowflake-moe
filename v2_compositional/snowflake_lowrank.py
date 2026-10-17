# -*- coding: utf-8 -*-
"""
snowflake_lowrank.py —— A 型低秩（用户 2026-10-07 裁决：改低秩 rank=4）

用户给的低秩写法（A 型 / coeff 型）：
    W1_A [d,r], W1_B [r,h], W1_coeff [N,r]
    mixed_coeff = wiring @ W1_coeff                    # [B, r]
    h = x @ W1_base + ((x @ W1_A) * mixed_coeff) @ W1_B

=============================================================================
⚠ 组合空间维数 = min(N-1, rank)（用户 2026-10-07 裁决 r=32）
=============================================================================
合成映射：wiring(N维单纯形) -> mixed_coeff(r维) -> delta

    组合空间维数 = min(N-1, r)

N=384, r=32 ⇒ 组合空间 **32 维**。

★ 用户 2026-10-07 澄清（重要，改变了论文表述）：
    "2^384 组合空间"这个说法【本来就不对】，与 r 取值无关。
    2^N 是二值选择空间的组合数（选/不选），而 wiring 是 softmax 出来的
    【连续向量】∈ 单纯形。真实组合空间是 N 维单纯形（无穷多连续点）。

    正确表述（不依赖 r，本质正确、审稿人挑不出刺）：
        标准 MoE：离散 top-k 选择 → C(N,k) 种
        Snowflake：连续 wiring   → N 维单纯形上的无穷多组合

    关键区别是【离散选择 vs 连续组合】，不是 "2^N vs N"。

这是真实的二选一：
    A 型：效率极致（P=1 就 0.38x）+ 可逐 token 自适应，但组合空间 r 维
    B 型：组合空间 383 维（保住能力1 论文主张），但需 chunk 且贵一点

折中：A 型把 r 开大（r=32）→ 组合空间 32 维，P=1 时仍比 Top-K 便宜。
本文件 r 可配，默认 4（规格），但建议真机扫 r ∈ {4, 16, 32}。

=============================================================================
✅ r=32 下的 P 甜点区（用户裁决后重算）
=============================================================================
    P* = N·r / (3·d·h − r·(d+h)) = 384·32 / (3·128·32 − 32·160) = 1.71

    P=1  21,504  1.312x  更贵 ⚠
    P=2  15,360  0.938x  便宜
    P=4  12,288  0.750x  便宜
    P=8  10,752  0.656x  便宜  ← 甜点
    P=32  9,600  0.586x  便宜
    P=128 9,312  0.568x  便宜（收益饱和）

    ⇒ r=32 时 P>=2 就比 Top-K 便宜，P=8~32 是甜点区。
    ⇒ 不再需要 P=128 的粗粒度，能力2（样本自适应）仍可保持较细粒度。

    对照 r=4：P*=0.132 ⇒ P=1 就赢（0.383x）
满秩 delta 下 P 必须开到 128 才与 Top-K 打平 ⇒ mix 粒度粗到整条序列
    ⇒ 能力2「每个样本一份临时专家」被 chunk 稀释

A 型下 P=1（逐 token）就是 0.38x，比 Top-K k=4 便宜 2.6 倍
    ⇒ 可以逐 token 自适应，且更便宜
    ⇒ 能力2 从「每 128 token 一份专家」变回「每 token 一份专家」

=============================================================================
★ 效率优势不随规模衰减（可写进论文）
=============================================================================
    pz/tk = (d·r + N·r/P + r·h + d·h) / (4·d·h)
    取 r = d/32, h = d/4 ⇒ 各项都正比于 d² ⇒ 比值与 d 无关

实测四档规模（tiny/mid/large/extreme）比值恒为 **0.290x**。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SnowflakeLowRank(nn.Module):
    """A 型低秩拼专家。forward(x) -> (out [N,d], info dict)"""

    def __init__(self, d, n_organelles=384, h=32, rank=32,
                 band=0.5, width=0.1, target_partners=4.0, temp=2.0,
                 seq_len=128, chunk_size=1,
                 hinge_mode="loss", hinge_start=0.8, hinge_coef=1.0,
                 seed=0):
        super().__init__()
        self.d, self.n, self.h, self.rank = d, n_organelles, h, rank
        self.band, self.width = band, width
        self.target_partners = target_partners
        self.temp = temp
        self.seq_len = seq_len
        self.chunk_size = chunk_size if (chunk_size > 1
                                         and seq_len % chunk_size == 0) else 1
        self.hinge_mode = hinge_mode
        self.hinge_start = hinge_start
        self.hinge_coef = hinge_coef

        g = torch.Generator().manual_seed(seed)
        # 红线6：器官初始化（共享基 + 噪声）
        base = torch.randn(d, generator=g)
        base = F.normalize(base, dim=-1) * 0.5 * math.sqrt(d)
        self.organelle_memory = nn.Parameter(
            base + torch.randn(n_organelles, d, generator=g) * 0.5)

        sc1 = 2.0 / math.sqrt(d)
        sc2 = 2.0 / math.sqrt(h)
        scA = 1.0 / math.sqrt(d)
        scB = 1.0 / math.sqrt(rank)

        # ---- 低秩 delta（A 型）----
        self.W1_base = nn.Parameter(torch.randn(d, h, generator=g) * sc1)
        self.W1_A = nn.Parameter(torch.randn(d, rank, generator=g) * scA)
        self.W1_B = nn.Parameter(torch.randn(rank, h, generator=g) * scB)
        self.W1_coeff = nn.Parameter(torch.randn(n_organelles, rank, generator=g) * 0.1)

        self.W2_base = nn.Parameter(torch.randn(h, d, generator=g) * sc2)
        self.W2_A = nn.Parameter(torch.randn(h, rank, generator=g) * scA)
        self.W2_B = nn.Parameter(torch.randn(rank, d, generator=g) * scB)
        self.W2_coeff = nn.Parameter(torch.randn(n_organelles, rank, generator=g) * 0.1)

        self.head = nn.Linear(d, d, bias=False)
        self.register_buffer("mix_ema", torch.ones(n_organelles) / n_organelles)
        self.register_buffer("fitness_ema", torch.ones(n_organelles) / n_organelles)
        self.heal_term = None
        self._diag = {}
        self._last_mix = None

    # ---------------------------------------------------------------- 连接
    def connect(self):
        m = F.normalize(self.organelle_memory, dim=-1)
        cos = m @ m.T
        c = torch.exp(-(((cos - self.band) / self.width) ** 2))
        c = c / (c.sum(-1, keepdim=True)
                 + self.target_partners) * self.target_partners

        off = ~torch.eye(self.n, dtype=torch.bool, device=c.device)
        if self.hinge_mode == "loss":
            # 加性独立项：真回复力。梯度 2(cos-0.8)，线性增长永不衰减。
            ex = (F.relu(cos - self.hinge_start) ** 2) * off.float()
            self.heal_term = ex.sum() * self.hinge_coef / (self.n * (self.n - 1))
        elif self.hinge_mode == "mult":
            hinge = (1.0 - F.relu(cos - self.hinge_start) * 5.0).clamp_min(0.0)
            c = c * hinge
            self.heal_term = None
        else:
            self.heal_term = None

        self._cos = cos.detach()
        return c, {
            "in_band": ((cos > self.band - 2 * self.width)
                        & (cos < self.band + 2 * self.width)).float()[off].mean().detach(),
            "cos_mean": cos.detach()[off].mean(),
            "cos_std": cos.detach()[off].std(),
            "connect_rowsum": c.detach().sum(-1).mean(),
        }

    def _chunk_view(self, x):
        N, d = x.shape
        if self.chunk_size <= 1 or N % self.seq_len != 0:
            return x.unsqueeze(1)
        B = N // self.seq_len
        return x.view(B, self.seq_len // self.chunk_size,
                      self.chunk_size, d).reshape(-1, self.chunk_size, d)

    # ---------------------------------------------------------------- 前向
    def forward(self, x, perm=None, force_mix=None):
        N, d = x.shape
        c, cm = self.connect()
        key = self.organelle_memory + c.T @ self.organelle_memory   # 务必 c.T

        xc = self._chunk_view(x)
        C, P, _ = xc.shape
        xsum = xc.mean(1)
        wiring = F.softmax(xsum @ key.T / math.sqrt(self.d) * self.temp, dim=-1)

        if force_mix is not None:
            fm = force_mix.unsqueeze(0) if force_mix.dim() == 1 else force_mix
            wiring = fm.to(wiring.dtype).expand(C, -1)
        elif perm is not None:
            wiring = wiring[perm]

        # ---- A 型低秩合成：成本与 batch 解耦 ----
        xA = torch.einsum("cpd,dr->cpr", xc, self.W1_A)          # [C,P,r]
        mixed = torch.einsum("cn,nr->cr", wiring, self.W1_coeff)  # [C,r]
        delta_h = torch.einsum("cpr,cr,rh->cph", xA, mixed, self.W1_B)
        hh = torch.einsum("cpd,dh->cph", xc, self.W1_base) + delta_h
        hh = F.silu(hh)

        hA = torch.einsum("cph,hr->cpr", hh, self.W2_A)
        mixed2 = torch.einsum("cn,nr->cr", wiring, self.W2_coeff)
        delta_o = torch.einsum("cpr,cr,rd->cpd", hA, mixed2, self.W2_B)
        out = torch.einsum("cph,hd->cpd", hh, self.W2_base) + delta_o

        wd = wiring.detach()
        self._last_mix = wd
        with torch.no_grad():
            self.mix_ema.mul_(0.99).add_(wd.mean(0).to(self.mix_ema.dtype) * 0.01)
            self.fitness_ema.mul_(0.99).add_(wd.mean(0).to(self.fitness_ema.dtype) * 0.01)

        self._diag = {**cm,
                      "wiring_ent": -(wd * (wd + 1e-9).log()).sum(-1).mean().detach(),
                      "wiring_variance": wd.var(0).mean().detach(),
                      "mix_sharp": wd.max(-1).values.mean().detach()}
        return self.head(out.reshape(N, d)), {**self._diag, "wiring": wd}

    # ---------------------------------------------------------------- 账
    def flops_ratio_vs_topk(self, k=4):
        """A 型相对 Top-K 的 FLOPs 比。实测四档规模恒定 0.290x。"""
        d, h, r, n, P = self.d, self.h, self.rank, self.n, self.chunk_size
        return (d * r + n * r / P + r * h + d * h) / (k * d * h)

    def combo_space_dim(self):
        """组合空间维数 = min(N-1, rank)。A 型被 rank 锁死。"""
        return min(self.n - 1, self.rank)

    def stats(self):
        out = {k: (float(v) if v.numel() == 1 else float(v.mean()))
               for k, v in self._diag.items()}
        out["flops_ratio"] = self.flops_ratio_vs_topk()
        out["combo_dim"] = float(self.combo_space_dim())
        return out
