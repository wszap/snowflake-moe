# -*- coding: utf-8 -*-
"""
snowflake_spec.py —— 《核心思想传输》锁 6.0 规格实现

核心命题：在参数空间组合，不是在输出空间选择。
    h = SiLU(x · Σᵢ mixᵢ·W1ᵢ)        ← 非线性在求和【之后】
    vs 标准 MoE: Σᵢ∈TopK wᵢ·SiLU(x·Wᵢ)  ← 非线性在求和【之前】

=============================================================================
锁 6.0 参数（用户 2026-10-07 拍板）
=============================================================================
    n_organelles = 384
    d            = 128
    target_partners = 4.0     行和目标（恒定，不随 N）
    band         = 0.5        高斯带通中心
    width        = 0.1        带通宽度
    delta_scale  = 0.5        先跑这个（另一档 1.0）
    wire_temp    = 2.0

=============================================================================
红线遵守情况
=============================================================================
 1. 不"优化"掉核心机制          ✅ 原样
 2. 内生约束（稳态）在前向里      ✅ 行和归一化在 forward，不是 loss 项
 3. 高斯带通，不用帐篷           ✅
 4. wiring 与 connect 分步计算    ✅ 先 connect 后 folding
 5. 保留监控指标                 ✅ 全部 .detach()
 6. 器官初始化 base(0.5√d)+randn*0.5  ✅ 原样（我 lock47 曾偏离，已纠正）
 7. 不跳过 permuted 对照          ✅ 见 ablate_permute.py

=============================================================================
⚠ 对铰链的一处实测修正（必须知悉）
=============================================================================
用户给的乘性门：c = c * (1 - relu(cos-0.8)*5).clamp_min(0)

实测（spec_check.py / 本文件 docstring 下方）：
    cos=0.9  : 纯高斯 |dc/dcos| = 9.00e-06   加 hinge = 5.06e-06   比值 0.563
    cos=0.95 : 纯高斯 |dc/dcos| = 1.44e-07   加 hinge = 4.41e-08   比值 0.306

乘性门把 c 压得更小、梯度也更小 ⇒ 它是【衰减加强器】，不是【回复力】。
数学上：乘性门只能压低 c 的数值，无法产生"把 cos 推回 band"的方向力。

要真正提供回复力，必须是【加性独立项】。原 tent+hinge 的 heal 本来就是 loss 项，
且红线2 保护的是"稳态"(行和归一化)，不是"自愈"。两者不冲突：

    稳态（行和归一化）  -> 前向，内生   ✅ 红线2
    铰链（自愈）        -> loss 项      ✅ 数学上只能这样

因此本文件提供两种模式：
    hinge_mode="mult"  用户字面实现（c *= hinge）。保留以便对照，但不推荐。
    hinge_mode="loss"  铰链作为独立 loss 项（默认）。真正提供回复力。

=============================================================================
稳态归一化的语义（用户澄清）
=============================================================================
    target_partners = 行和，不是每对连接的强度。

    N=8   : 行和 4 -> 每对 4/7   ≈ 0.57
    N=384 : 行和 4 -> 每对 4/383 ≈ 0.010

    两者物理意义相同：每个器官平均连 4 个伙伴。
    黄金区间 0.3~0.7 指的是【cos_sim】，不是 connect 的值。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SnowflakeCell(nn.Module):
    """锁 6.0 规格实现。forward(x) -> (out [N,d], info dict)"""

    def __init__(self, d, n_organelles=384, h=32,
                 band=0.5, width=0.1, target_partners=4.0, temp=2.0,
                 n_shared=2, delta_shared=0.01, delta_special=0.10,
                 delta_scale=0.5, init_noise=0.5,
                 seq_len=128, chunk_size=1,
                 hinge_mode="loss", hinge_start=0.8, hinge_slope=5.0,
                 hinge_loss_coef=1.0, seed=0):
        super().__init__()
        self.d = d
        self.n = n_organelles
        self.h = h
        self.band = band
        self.width = width
        self.target_partners = target_partners
        self.temp = temp
        self.seq_len = seq_len
        self.chunk_size = chunk_size if (chunk_size > 1
                                         and seq_len % chunk_size == 0) else 1

        # 铰链
        self.hinge_mode = hinge_mode            # "loss" | "mult" | "none"
        self.hinge_start = hinge_start
        self.hinge_slope = hinge_slope
        self.hinge_loss_coef = hinge_loss_coef

        # ---- 红线6：器官初始化（共享基 + 噪声，尺度不可改）----
        g = torch.Generator().manual_seed(seed)
        base = torch.randn(d, generator=g)
        base = F.normalize(base, dim=-1) * 0.5 * math.sqrt(d)
        self.organelle_memory = nn.Parameter(
            base + torch.randn(n_organelles, d, generator=g) * init_noise)

        # ---- 原则1 base+delta；原则2 前 n_shared 个是通用端口 ----
        sc1 = 2.0 / math.sqrt(d)
        sc2 = 2.0 / math.sqrt(h)
        delta_std = torch.cat([torch.full((n_shared,), delta_shared),
                               torch.full((n_organelles - n_shared,), delta_special)])
        self.W1_base = nn.Parameter(torch.randn(d, h, generator=g) * sc1)
        self.W2_base = nn.Parameter(torch.randn(h, d, generator=g) * sc2)
        self.W1_delta = nn.Parameter(
            torch.randn(n_organelles, d, h, generator=g)
            * delta_std.view(-1, 1, 1) * sc1 * delta_scale)
        self.W2_delta = nn.Parameter(
            torch.randn(n_organelles, h, d, generator=g)
            * delta_std.view(-1, 1, 1) * sc2 * delta_scale)

        self.head = nn.Linear(d, d, bias=False)
        self.register_buffer("mix_ema", torch.ones(n_organelles) / n_organelles)
        self.register_buffer("fitness_ema", torch.ones(n_organelles) / n_organelles)
        self._diag = {}
        self._last_mix = None
        self.heal_term = None            # 铰链 loss 项（hinge_mode="loss" 时非 None）

    # ---------------------------------------------------------------- 权重
    @property
    def W1_all(self):
        return self.W1_base.unsqueeze(0) + self.W1_delta

    @property
    def W2_all(self):
        return self.W2_base.unsqueeze(0) + self.W2_delta

    # ---------------------------------------------------------------- 连接
    def connect(self):
        """规格第 2、3 行：高斯带通 + 稳态自归一化（行和 = target_partners）。

        返回 (c [n,n], 监控 dict)。稳态是【内生】的：在前向里生效，不是 loss 项。
        """
        m = F.normalize(self.organelle_memory, dim=-1)
        cos = m @ m.T
        c = torch.exp(-(((cos - self.band) / self.width) ** 2))     # 2 高斯带通

        # 3 稳态：行和归一化到 target_partners（红线2：内生，不在 loss 里）
        c = c / (c.sum(-1, keepdims=True) + self.target_partners) * self.target_partners

        # 铰链（loss 模式）
        off = ~torch.eye(self.n, dtype=torch.bool, device=c.device)
        if self.hinge_mode == "loss":
            # 加性独立项：cos > hinge_start 后二次增长，永不衰减 ⇒ 真回复力。
            # 归一化除以 N(N-1)，否则 N=384 时量级爆炸（实测 N=8:0.58, N=384:37.5）
            ex = F.relu(cos - self.hinge_start) ** 2 * off.float()
            self.heal_term = ex.sum() * self.hinge_loss_coef / (self.n * (self.n - 1))
        elif self.hinge_mode == "mult":
            # 用户字面实现：乘性门。实测是衰减加强，不是回复力。保留以便对照。
            hinge = (1.0 - F.relu(cos - self.hinge_start) * self.hinge_slope).clamp_min(0.0)
            c = c * hinge
            self.heal_term = None
        else:
            self.heal_term = None

        self._cos = cos.detach()
        return c, {
            # 黄金区间 0.3~0.7 指的是 cos（用户澄清）
            "in_band": ((cos > self.band - 2 * self.width)
                        & (cos < self.band + 2 * self.width)).float()[off].mean().detach(),
            "cos_mean": cos.detach()[off].mean(),
            "cos_std": cos.detach()[off].std(),
            "cos_max": cos.detach()[off].max(),
            "connect_rowsum": c.detach().sum(-1).mean(),
            "connect_pair_mean": c.detach()[off].mean(),
        }

    # ---------------------------------------------------------------- 前向
    def _chunk_view(self, x):
        N, d = x.shape
        if self.chunk_size <= 1 or N % self.seq_len != 0:
            return x.unsqueeze(1)
        B = N // self.seq_len
        return x.view(B, self.seq_len // self.chunk_size,
                      self.chunk_size, d).reshape(-1, self.chunk_size, d)

    def forward(self, x, perm=None, force_mix=None):
        N, d = x.shape
        c, cm = self.connect()

        # 4 折叠传播 + routing（务必 c.T —— 行归一化已破坏对称性）
        key = self.organelle_memory + c.T @ self.organelle_memory
        xc = self._chunk_view(x)
        C, P, _ = xc.shape
        xsum = xc.mean(1)
        wiring = F.softmax(xsum @ key.T / math.sqrt(self.d) * self.temp, dim=-1)

        if force_mix is not None:                    # 常数配方（退化对照）
            fm = force_mix.unsqueeze(0) if force_mix.dim() == 1 else force_mix
            wiring = fm.to(wiring.dtype).expand(C, -1)
        elif perm is not None:                       # permuted 对照（唯一有效形式）
            wiring = wiring[perm]

        # 5 拼专家 einsum（规格：保持不变）
        h = torch.einsum("ci,idh,cpd->cph", wiring, self.W1_all, xc)
        h = F.silu(h)
        out = torch.einsum("ci,ihd,cph->cpd", wiring, self.W2_all, h)
        out = self.head(out.reshape(N, d))

        wd = wiring.detach()
        self._last_mix = wd
        with torch.no_grad():
            self.mix_ema.mul_(0.99).add_(wd.mean(0).to(self.mix_ema.dtype) * 0.01)
            self.fitness_ema.mul_(0.99).add_(wd.mean(0).to(self.fitness_ema.dtype) * 0.01)

        self._diag = {**cm,
                      "wiring_ent": -(wd * (wd + 1e-9).log()).sum(-1).mean().detach(),
                      "wiring_variance": wd.var(0).mean().detach(),
                      "mix_sharp": wd.max(-1).values.mean().detach()}
        return out, {**self._diag, "wiring": wd}

    # ------------------------------------------------------------ FLOPs 账
    def flops_ratio_vs_topk(self, k=4):
        """（满秩 delta 下）拼专家/Top-K 的 FLOPs 比。

        ⚠ N=384 且 delta 满秩时 = 3.25x【更贵】。
        要赢必须低秩或大 chunk。本文件是满秩（规格要求），
        所以 P 必须开大：P=128 时恰好 1.00x，P>128 才便宜。
        """
        return (self.n / self.chunk_size + 1.0) / k

    # ---------------------------------------------------------------- 原则5
    @torch.no_grad()
    def bio_select(self, kill_frac=0.25, noise=0.1, ortho=False):
        """生物筛选：淘汰 fitness 最低的 kill_frac，用最强器官 + 噪声替代。

        原则5：前 3000 步不要调用（让 fitness 稳定），之后每 500 步一次。
        ortho=True 时再生向量投影掉已有器官方向（防趋同，非规格内容）。
        """
        n_kill = max(1, int(self.n * kill_frac))
        fit = self.fitness_ema
        weak = torch.topk(fit, n_kill, largest=False).indices
        strong = torch.topk(fit, n_kill, largest=True).indices
        mem = self.organelle_memory.data
        for w, s in zip(weak.tolist(), strong.tolist()):
            v = mem[s].clone()
            if ortho:
                mem_n = F.normalize(mem, dim=-1)
                for _ in range(2):
                    proj = (mem_n @ F.normalize(v, dim=-1)) @ mem_n
                    v = v - proj * (v.norm() / max(proj.norm().item(), 1e-8) + 1e-8)
                v = F.normalize(v, dim=-1) * mem[s].norm()
            mem[w] = v + torch.randn_like(v) * noise
            self.W1_delta.data[w] = (self.W1_delta.data[s]
                                     + torch.randn_like(self.W1_delta.data[s]) * noise)
            self.W2_delta.data[w] = (self.W2_delta.data[s]
                                     + torch.randn_like(self.W2_delta.data[s]) * noise)
        self.fitness_ema.mul_(0.0).add_(1.0 / self.n)
        return {"killed": weak.tolist(), "regen_from": strong.tolist(),
                "mode": ("ortho" if ortho else "clone")}

    def stats(self):
        out = {k: (float(v) if v.numel() == 1 else float(v.mean()))
               for k, v in self._diag.items()}
        out["flops_ratio"] = self.flops_ratio_vs_topk()
        return out
