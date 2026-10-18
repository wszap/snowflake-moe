"""
真·拼专家（v8）——“核架选谁，忆点载荷，拼出更宽的专家”

============================================================================
                        与旧版（snowflake_B.py）的根本区别
============================================================================

 旧版（先混合参数 → 单个共享非线性）
     U1  = Σᵢ wᵢ · W1_U[i]                     形状永远是 [d, rank]
     hh  = silu( x@W1_base + (x@U1)@W1_V )     只有一个共享 silu，宽 h=32
     out = hh@W2_base + (hh@U2)@W2_V

   ⇒ U1 是 [d,rank] 矩阵，秩 ≤ rank。混合 1 个和混合 384 个，秩都是 4。
     “拼”发生在瓶颈内部 ⇒ 拼 k 个 = 拼 1 个。这就是“32 人挤 4 人道”。

 新版（各自非线性 → 加权求和，等价于隐层拼接）
     hᵢ  = silu( x @ E1ᵢ )        每个忆点独立做一次非线性，宽 h_cell
     out = base(x) + g · Σᵢ wᵢ · ( hᵢ @ E2ᵢ )

   ⇒ 数学恒等式：
        Σᵢ wᵢ σ(xE1ᵢ)E2ᵢ  ≡  [σ(xE1₁), …, σ(xE1_k)] @ blkdiag(wᵢE2ᵢ)
     即“加权求和” ≡ “把 k 段隐层拼起来再投影”，拼后宽度 = k × h_cell。
     拼 k 个 = k 倍容量，随 k 线性增长。（verify 第①②条有数值证明）

============================================================================
                              三条主张如何兑现
============================================================================
 自主选拼 : wiring 完全沿用内生机制（忆点匹配 + 核架），不引入任何外部路由器。
           不同输入 → 不同忆点被选中 → 拼出不同的专家（verify 第⑥条）。
 真扩容   : 拼后等效隐层宽度 = k × h_cell，随 k 线性增长（verify 第①②条）。
 高效     : 只算 top-k 个专家。每 token FLOPs = 4·k·d·h_cell，
           k=16,h_cell=32 时与 dense 持平，但总容量是 dense 的 24 倍（第⑤条）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComposableExpert(nn.Module):
    """drop-in 替代 SnowflakeB：forward(x, perm, force_mix) -> (out, info)"""

    def __init__(self, d, n_organelles=384, h=32, rank=4,
                 h_cell=32, k=8, renorm=True,
                 band=0.5, width=0.1, target_partners=4.0, temp=2.0,
                 seq_len=128, chunk_size=32,
                 hinge_mode="loss", hinge_start=0.8, hinge_coef=1.0,
                 hinge_low=0.3, hinge_low_coef=1.0, use_hinge_low=False,
                 sharp_warn=0.7, log_sharp=False,
                 min_eff_organs=0, seed=0, mem_budget_mb=256.0):
        super().__init__()
        self.d, self.n, self.h = d, n_organelles, h
        self.rank = rank                    # 仅为兼容旧签名，新版不再限制容量
        self.h_cell, self.k = h_cell, min(k, n_organelles)
        self.renorm = renorm
        self.band, self.width = band, width
        self.target_partners = target_partners
        self.temp = temp
        self.seq_len = seq_len
        self.chunk_size = chunk_size if (chunk_size > 1
                                         and seq_len % chunk_size == 0) else 1
        self.hinge_mode = hinge_mode
        self.hinge_start, self.hinge_coef = hinge_start, hinge_coef
        self.hinge_low, self.hinge_low_coef = hinge_low, hinge_low_coef
        self.use_hinge_low = use_hinge_low
        self.sharp_warn, self.log_sharp = sharp_warn, log_sharp
        self.min_eff_organs = int(min_eff_organs)
        # 显存安全阀：一次性 gather 会形成 [C,k,d,h_cell]。
        # 超预算时自动改逐 slot 计算（峰值降为 1/k），使 chunk=1 也能跑。
        self.mem_budget_mb = float(mem_budget_mb)

        g = torch.Generator().manual_seed(seed)

        # ---- 忆点（方向载体，沿用红线6的原初始化，尺度不可改）----
        base = torch.randn(d, generator=g)
        base = F.normalize(base, dim=-1) * 0.5 * math.sqrt(d)
        self.organelle_memory = nn.Parameter(
            base + torch.randn(n_organelles, d, generator=g) * 0.5)

        # ---- 共享基底：所有 token 都经过的公共通道（= v4_shared 证明有效的那一层）----
        self.W1_base = nn.Parameter(torch.randn(d, h, generator=g)
                                    * (2.0 / math.sqrt(d)))
        self.W2_base = nn.Parameter(torch.randn(h, d, generator=g)
                                    * (2.0 / math.sqrt(h)))
        self.b2 = nn.Parameter(torch.zeros(d))

        # ---- 忆点载荷：每个忆点一个独立小 FFN（d → h_cell → d）----
        s1 = 1.0 / math.sqrt(d)
        s2 = 1.0 / math.sqrt(h_cell)
        self.E1 = nn.Parameter(torch.randn(n_organelles, d, h_cell,
                                           generator=g) * s1)
        self.E2 = nn.Parameter(torch.randn(n_organelles, h_cell, d,
                                           generator=g) * s2)

        # ---- 零门：起步 g=0 ⇒ 严格恒等；且 E1/E2 随机 ⇒ g 有梯度，不会死锁 ----
        self.gate = nn.Parameter(torch.zeros(1))

        self.head = nn.Linear(d, d, bias=False)
        self.register_buffer("mix_ema", torch.ones(n_organelles) / n_organelles)
        self.register_buffer("fitness_ema", torch.ones(n_organelles) / n_organelles)
        self.heal_term = None
        self._diag = {}
        self._last_mix = None

    # ------------------------------------------------------------ 连接
    def connect(self):
        m = F.normalize(self.organelle_memory, dim=-1)
        cos = m @ m.T
        c = torch.exp(-(((cos - self.band) / self.width) ** 2))
        c = c / (c.sum(-1, keepdim=True)
                 + self.target_partners) * self.target_partners
        off = ~torch.eye(self.n, dtype=torch.bool, device=c.device)
        self.heal_term = None
        if self.hinge_mode == "loss":
            ex = (F.relu(cos - self.hinge_start) ** 2) * off.float()
            self.heal_term = ex.sum() * self.hinge_coef / (self.n * (self.n - 1))
            if self.use_hinge_low:
                exl = (F.relu(self.hinge_low - cos) ** 2) * off.float()
                self.heal_term = (self.heal_term
                                  + exl.sum() * self.hinge_low_coef
                                  / (self.n * (self.n - 1)))
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

    # ------------------------------------------------------------ 前向
    def forward(self, x, perm=None, force_mix=None):
        N, d = x.shape
        c, cm = self.connect()
        key = self.organelle_memory + c.T @ self.organelle_memory

        xc = self._chunk_view(x)                      # [C, P, d]
        C, P, _ = xc.shape
        xsum = xc.mean(1)
        wiring = F.softmax(xsum @ key.T / math.sqrt(self.d) * self.temp, dim=-1)

        if self.min_eff_organs > 0:
            wiring = self.enforce_min_effective(wiring, self.min_eff_organs)
        if force_mix is not None:
            fm = force_mix.unsqueeze(0) if force_mix.dim() == 1 else force_mix
            wiring = fm.to(wiring.dtype).expand(C, -1)
        elif perm is not None:
            wiring = wiring[perm]

        # ---------- ① 自主选：top-k 忆点（内生 wiring，无外部路由器）----------
        topw, topi = torch.topk(wiring, self.k, dim=-1)       # [C, k]
        if self.renorm:
            topw = topw / topw.sum(-1, keepdim=True).clamp_min(1e-9)

        # ---------- ② 共享基底（公共通道）----------
        out = F.silu(xc @ self.W1_base) @ self.W2_base + self.b2   # [C,P,d]

        # ---------- ③ 拼：各自非线性 → 加权求和（≡ 隐层拼接）----------
        # 显存安全阀：need 是一次性 gather 的 [C,k,d,h_cell] 占用(MB)
        need = C * self.k * d * self.h_cell * 4.0 / (1024 ** 2)
        if need > self.mem_budget_mb:
            # 逐 slot：峰值 [C,d,h_cell]，为一次性版的 1/k
            comp = torch.zeros_like(out)
            for _i in range(self.k):
                E1i = self.E1.index_select(0, topi[:, _i])    # [C,d,h_cell]
                E2i = self.E2.index_select(0, topi[:, _i])    # [C,h_cell,d]
                _h = F.silu(torch.bmm(xc, E1i))               # [C,P,h_cell]
                comp = comp + topw[:, _i][:, None, None] * torch.bmm(_h, E2i)
            self._lowmem_used = True
        else:
            E1s = self.E1[topi]                               # [C,k,d,h_cell]
            E2s = self.E2[topi]                               # [C,k,h_cell,d]
            hh = torch.matmul(xc.unsqueeze(1), E1s)           # [C,k,P,h_cell]
            hh = F.silu(hh)
            oo = torch.matmul(hh, E2s)                        # [C,k,P,d]
            comp = torch.einsum('ck,ckpd->cpd', topw, oo)     # [C,P,d]
            self._lowmem_used = False
        out = out + self.gate * comp

        # ---------- 诊断（沿用旧版口径，保证指标可比）----------
        wd = wiring.detach()
        self._last_mix = wd
        with torch.no_grad():
            self.mix_ema.mul_(0.99).add_(wd.mean(0).to(self.mix_ema.dtype) * 0.01)
            self.fitness_ema.mul_(0.99).add_(wd.mean(0).to(self.fitness_ema.dtype) * 0.01)
        sharp = wd.max(-1).values.mean().detach()
        eff_n = 1.0 / (wd * wd).sum(-1).mean().detach()
        _mean = 1.0 / self.n
        var_max = (1.0 / self.n) * (1 - _mean) ** 2 + \
                  ((self.n - 1) / self.n) * _mean ** 2
        wvar = wd.var(0).mean().detach()

        # 被选中忆点的多样性（体现“自发分工”）
        uniq = torch.unique(topi).numel()

        self._diag = {**cm,
                      "wiring_ent": -(wd * (wd + 1e-9).log()).sum(-1).mean().detach(),
                      "wiring_variance": wvar,
                      "wiring_var_norm": wvar / float(var_max),
                      "eff_organs": eff_n,
                      "mix_sharp": sharp,
                      "gate": self.gate.detach().mean(),
                      "uniq_experts": float(uniq),
                      "composed_width": float(self.k * self.h_cell),
                      "lowmem_used": getattr(self, "_lowmem_used", False),
                      "mem_need_mb": float(need)}
        return self.head(out.reshape(N, d)), {**self._diag, "wiring": wd}

    # ------------------------------------------------- 最低有效器官数约束
    @staticmethod
    def enforce_min_effective(w, K):
        """把 wiring 与均匀分布混合，使有效器官数 1/Σw² 精确等于 K。"""
        n = w.shape[-1]
        S2 = (w * w).sum(-1)
        A = S2 - 1.0 / n
        C = S2 - 1.0 / K
        a = torch.where(A.abs() > 1e-12,
                        (A - torch.sqrt((A * A - A * C).clamp_min(0.0))) / A.clamp_min(1e-12),
                        torch.zeros_like(S2))
        a = a.clamp(0.0, 1.0).unsqueeze(-1)
        return (1 - a) * w + a / n

    # ------------------------------------------------------------ 账本
    def flops_per_token(self):
        """每 token 的乘加次数（只算激活的 k 个专家 + 共享基底）。"""
        base = 2 * self.d * self.h * 2            # W1_base + W2_base
        experts = self.k * (2 * self.d * self.h_cell * 2)
        return base + experts

    def expert_params(self):
        """忆点库总容量（不激活也占显存，这是 MoE 的“总容量”）。"""
        return self.n * (self.d * self.h_cell + self.h_cell * self.d)

    def stats(self):
        return {
            "k": self.k, "h_cell": self.h_cell,
            "composed_width": self.k * self.h_cell,
            "flops_per_token": self.flops_per_token(),
            "expert_total_params": self.expert_params(),
            "gate": float(self.gate.detach().mean()),
        }
