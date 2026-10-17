# -*- coding: utf-8 -*-
"""
snowflake_B.py —— B 型低秩（每器官独立因子），用户 2026-10-07 裁决：停用 A 型，改 B 型

=============================================================================
B 型写法（factor 型）
=============================================================================
    W1_U [N,d,r]   每个器官一个独立低秩因子
    W1_V [r,h]     共享重构基

    U_wired = einsum('cn,ndr->cdr', wiring, W1_U)      # [C,d,r]  合成，与 batch 无关
    xA      = einsum('cpd,cdr->cpr', xc, U_wired)      # [C,P,r]
    h       = xc @ W1_base + xA @ W1_V                 # [C,P,h]

=============================================================================
★ 为什么 B 型严格优于 A 型（实测）
=============================================================================
组合空间维数 = min(N-1, d·r)     ← B 型，r 只通过 d·r 起作用
组合空间维数 = min(N-1, r)       ← A 型，被 r 直接锁死

小规模实测（d=64,h=16）：
    N=16 r=4: A型=4 维   B型=15 维（理论 min(15,256)=15）✅
    N=64 r=4: A型=4 维   B型=61 维（理论 min(63,256)=63）✅
    N=64 r=8: A型=8 维   B型=62 维

⇒ **r=4 时 d·r = 512 已 > N-1 = 383，B 型直接拿到满维 383。**
   r 不需要开大。这与 A 型完全相反（A 型要 32 维就得 r=32）。

FLOPs 对比（N=384, d=128, h=32）：
    B型 r=4  P=32  →  10,880  →  0.664x  →  **383 维**
    A型 r=32 P=8   →  10,752  →  0.656x  →  **32 维**

    ★ 几乎相同的 FLOPs（差 1.2%），组合空间从 32 维涨到 383 维。
      B 型严格占优，无代价。

=============================================================================
⚠ B 型的 r 不能开大（与 A 型相反）
=============================================================================
    r=4  P=32 → 0.664x ✅
    r=8  P=32 → 1.078x ⚠
    r=16 P=32 → 1.906x ❌
    r=32 P=128→ 1.312x ❌（即使 P 拉满仍更贵）

    合成项 N·d·r/P 随 r 线性涨，而 A 型只有 N·r/P（差 d=128 倍）。
    ⇒ B 型的效率全靠【小 r + 大 P】。默认 r=4, P=32。

    若嫌 P=32 粒度太粗：P 越大越便宜，同时组合空间不变（383 维恒成立）。
    P=64 → 0.477x；P=128 → 0.383x。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SnowflakeB(nn.Module):
    """B 型低秩拼专家。forward(x) -> (out [N,d], info dict)"""

    def __init__(self, d, n_organelles=384, h=32, rank=4,
                 band=0.5, width=0.1, target_partners=4.0, temp=2.0,
                 seq_len=128, chunk_size=32,
                 hinge_mode="loss", hinge_start=0.8, hinge_coef=1.0,
                 hinge_low=0.3, hinge_low_coef=1.0, use_hinge_low=False,
                 sharp_warn=0.7, log_sharp=False,
                 min_eff_organs=0, seed=0):
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
        # ---- 双侧回复力（默认关闭，等 cos 漂移方向确认后再开）----
        self.hinge_low = hinge_low
        self.hinge_low_coef = hinge_low_coef
        self.use_hinge_low = use_hinge_low
        # ---- one-hot 告警 ----
        self.sharp_warn = sharp_warn
        self.log_sharp = log_sharp
        # ---- 内生最低有效器官数约束（默认关闭，0=不启用）----
        self.min_eff_organs = int(min_eff_organs)

        g = torch.Generator().manual_seed(seed)
        # 红线6：器官初始化（共享基 + 噪声，尺度不可改）
        base = torch.randn(d, generator=g)
        base = F.normalize(base, dim=-1) * 0.5 * math.sqrt(d)
        self.organelle_memory = nn.Parameter(
            base + torch.randn(n_organelles, d, generator=g) * 0.5)

        sc1 = 2.0 / math.sqrt(d)
        sc2 = 2.0 / math.sqrt(h)
        scU = 1.0 / math.sqrt(d)
        scV = 1.0 / math.sqrt(rank)

        # ---- B 型：每器官独立低秩因子 ----
        self.W1_base = nn.Parameter(torch.randn(d, h, generator=g) * sc1)
        self.W1_U = nn.Parameter(torch.randn(n_organelles, d, rank, generator=g) * scU)
        self.W1_V = nn.Parameter(torch.randn(rank, h, generator=g) * scV)

        self.W2_base = nn.Parameter(torch.randn(h, d, generator=g) * sc2)
        self.W2_U = nn.Parameter(torch.randn(n_organelles, h, rank, generator=g) * scU)
        self.W2_V = nn.Parameter(torch.randn(rank, d, generator=g) * scV)

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
        c = torch.exp(-(((cos - self.band) / self.width) ** 2))     # 2 高斯带通
        # 3 稳态：行和归一化到 target_partners（红线2：内生，前向里生效）
        c = c / (c.sum(-1, keepdims=True)
                 + self.target_partners) * self.target_partners

        off = ~torch.eye(self.n, dtype=torch.bool, device=c.device)
        if self.hinge_mode == "loss":
            # 加性独立项：真回复力，梯度 2(cos-0.8) 线性增长永不衰减
            ex = (F.relu(cos - self.hinge_start) ** 2) * off.float()
            self.heal_term = ex.sum() * self.hinge_coef / (self.n * (self.n - 1))
            # 下侧回复力：cos < hinge_low 时推回（默认关闭）
            if self.use_hinge_low:
                exl = (F.relu(self.hinge_low - cos) ** 2) * off.float()
                self.heal_term = (self.heal_term
                                  + exl.sum() * self.hinge_low_coef
                                  / (self.n * (self.n - 1)))
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

        # ---- 内生最低有效器官数约束：保证每样本至少真正用到 K 个器官 ----
        # 解决 sharp→1（384 选 1）的药方。见 D-005。
        if self.min_eff_organs > 0:
            wiring = self.enforce_min_effective(wiring, self.min_eff_organs)

        if force_mix is not None:
            fm = force_mix.unsqueeze(0) if force_mix.dim() == 1 else force_mix
            wiring = fm.to(wiring.dtype).expand(C, -1)
        elif perm is not None:
            wiring = wiring[perm]

        # ---- B 型合成：先合参数，再前向一次（成本与 batch 解耦）----
        U1 = torch.matmul(wiring, self.W1_U.view(self.n, -1)).view(-1, self.d, self.rank)          # [C,d,r]
        xA = torch.bmm(xc, U1)                    # [C,P,r]
        hh = (torch.matmul(xc, self.W1_base)
              + torch.matmul(xA, self.W1_V))
        hh = F.silu(hh)

        U2 = torch.matmul(wiring, self.W2_U.view(self.n, -1)).view(-1, self.h, self.rank)          # [C,h,r]
        hA = torch.bmm(hh, U2)                    # [C,P,r]
        out = (torch.matmul(hh, self.W2_base)
               + torch.matmul(hA, self.W2_V))

        wd = wiring.detach()
        self._last_mix = wd
        with torch.no_grad():
            self.mix_ema.mul_(0.99).add_(wd.mean(0).to(self.mix_ema.dtype) * 0.01)
            self.fitness_ema.mul_(0.99).add_(wd.mean(0).to(self.fitness_ema.dtype) * 0.01)

        sharp = wd.max(-1).values.mean().detach()
        # 有效器官数 = 1/Σw²。这是跨 N 可比的核心量（见 L-023）
        eff_n = 1.0 / (wd * wd).sum(-1).mean().detach()
        # 归一化方差：wiring_var 的绝对值随 N 变（N=384 的 one-hot 上限仅
        # 0.0026），0.05 门槛数学上不可达。改用 var / var_max（见 L-023）。
        _mean = 1.0 / self.n
        var_max = (1.0 / self.n) * (1 - _mean) ** 2 + \
                  ((self.n - 1) / self.n) * _mean ** 2
        wvar = wd.var(0).mean().detach()
        self._diag = {**cm,
                      "wiring_ent": -(wd * (wd + 1e-9).log()).sum(-1).mean().detach(),
                      "wiring_variance": wvar,
                      "wiring_var_norm": wvar / float(var_max),
                      "eff_organs": eff_n,
                      "mix_sharp": sharp}
        if self.log_sharp and float(sharp) > self.sharp_warn:
            print(f"[one-hot warning] mix_sharp={float(sharp):.3f} "
                  f"> {self.sharp_warn} ⇒ wiring 接近 one-hot，"
                  f"『连续组合』叙事可能不成立。先查数据分离度(见 --sep)，再改 temp。", flush=True)
        return self.head(out.reshape(N, d)), {**self._diag, "wiring": wd}

    # ------------------------------------------------- 最低有效器官数约束
    @staticmethod
    def enforce_min_effective(w, K):
        """把 wiring 与均匀分布混合，使有效器官数 1/Σw² 精确等于 K。

            w_a = (1-a)·w + a·(1/N)

        解 a：Σw_a² = (1-a)²S₂ + (2a-a²)/N = 1/K
            令 A = S₂ - 1/N, C = S₂ - 1/K
            A·a² - 2A·a + C = 0
            ⇒ (a-1)² = (A-C)/A = (1/K - 1/N)/(S₂ - 1/N)
            ⇒ a = 1 - sqrt((1/K - 1/N)/(S₂ - 1/N))

        ★ 性质（均已数值验证，见 verify_min_eff.py）
        1. **单边**：S₂ ≤ 1/K（已够分散）时 a ≤ 0 ⇒ clamp 到 0，完全不动
        2. **精确**：解析解，无迭代，梯度可回传
        3. **内生**：在前向里生效，不是 loss 惩罚（符合红线 2 精神）
        4. **每样本独立**：坍缩的样本多掺，已组合好的样本不动
        5. Cauchy-Schwarz 保证 S₂ ≥ 1/N ⇒ 分母恒非负，无除零风险
        """
        N = w.shape[-1]
        inv_N = 1.0 / N
        S2 = (w * w).sum(-1, keepdim=True)
        num = 1.0 / K - inv_N                       # > 0 只要 K < N
        den = (S2 - inv_N).clamp_min(1e-8)          # ≥ 0 by Cauchy-Schwarz
        ratio = (num / den).clamp_min(1e-8)
        a = (1.0 - ratio.sqrt()).clamp(0.0, 1.0)
        return (1.0 - a) * w + a * inv_N

    # ---------------------------------------------------------------- 账
    def flops_ratio_vs_topk(self, k=4):
        """B 型相对 Top-K 的 FLOPs 比。默认 r=4,P=32 → 0.664x。"""
        d, h, r, n, P = self.d, self.h, self.rank, self.n, self.chunk_size
        return (n * d * r / P + d * r + r * h + d * h) / (k * d * h)

    def combo_space_dim(self):
        """组合空间维数 = min(N-1, d·rank)。r=4 时 = 383（满维）。"""
        return min(self.n - 1, self.d * self.rank)

    def stats(self):
        out = {k: (float(v) if v.numel() == 1 else float(v.mean()))
               for k, v in self._diag.items()}
        out["flops_ratio"] = self.flops_ratio_vs_topk()
        out["combo_dim"] = float(self.combo_space_dim())
        return out
