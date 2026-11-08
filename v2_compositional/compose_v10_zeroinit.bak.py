# -*- coding: utf-8 -*-
"""
paperplaneMoe / step25 —— 真·拼专家 cell（v10）

来源：本仓库 v2_compositional/composable_expert.py (v8 真拼法)。
本文件只做「接进主训练」所需的适配，不改 v8 的数学定义。

与 snowflake_B.py（B5 当前用的旧拼法）的根本区别
------------------------------------------------
  旧  U1 = Σ_i w_i W1_U[i]  → 单一共享 silu，秩 ≤ rank=4
      ⇒ 拼 k 个 = 拼 1 个（"32 人挤 4 人道"）

  新  out = base(x) + g · Σ_i w_i · silu(x @ E1_i) @ E2_i
      ≡ [silu(xE1_1),...,silu(xE1_k)] @ blkdiag(w_i E2_i)
      ⇒ 等效隐层宽度 = k × h_cell，随 k 线性增长

保留的红线（与仓库既有结论一致）
  · 无外部 router，wiring 由忆点内生；全 n 权重恒非零（拼，不是选）
  · top-k 只是计算稀疏；eff / var_norm / mix_sharp 一律用【全 n】wiring 算
  · 新细胞用正锥份额 _split_child_w 追加，不进 softmax 抢名额
  · 旧 c 块冻结；E1/E2 新行零初始化 —— 不复制母体已学知识
  · gate = 0 起步严格恒等；heal_term 在冻结期置 None

接口与 snowflake_B.SnowflakeB 一致：
  forward(x, perm, force_mix) -> (out, info)
  属性：n / d / organelle_memory / _last_mix / _diag / heal_term /
        mix_ema / fitness_ema / connect() / heal_sum() 可用
  分裂钩子：_split_c / _split_n0 / _split_child_w / expand()
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from composable_expert import ComposableExpert


class ComposableExpertV10(ComposableExpert):
    """drop-in 替代 SnowflakeB；多出 wiring_mod 与分裂钩子。"""

    def __init__(self, *args, wiring_mod="none", **kwargs):
        super().__init__(*args, **kwargs)
        assert wiring_mod in ("none", "norm", "center"), wiring_mod
        self.wiring_mod = wiring_mod
        self._split_c = None
        self._split_n0 = 0
        self._split_child_w = None
        self.wiring_mode = "soft"
        self.heal_term = None

    # ---------------------------------------------------------------- 连接
    def connect(self):
        """冻结期直接返回旧 c（旧块原封不动），并关掉键长回复力。"""
        if self._split_c is not None:
            self.heal_term = None
            z = torch.zeros((), device=self.organelle_memory.device)
            return self._split_c, {
                "in_band": z, "cos_mean": z, "cos_std": z,
                "connect_rowsum": z,
            }
        return super().connect()

    # ---------------------------------------------------------------- 前向
    def forward(self, x, perm=None, force_mix=None):
        N, d = x.shape
        c, cm = self.connect()
        m_all = self.organelle_memory                       # [n_all, d]
        key = m_all + c.T @ m_all                          # 忆点互相关联
        if self._split_n0 > 0:
            key = key[:self._split_n0]                     # 新细胞不抢名额

        xc = self._chunk_view(x)                           # [C, P, d]
        C, P, _ = xc.shape
        xsum = xc.mean(1)
        # 输入调制（针对 RESULTS.md 里 H2「按输入组合」FAIL 的可选实验变量）
        if self.wiring_mod == "norm":
            xsum = F.normalize(xsum, dim=-1)
        elif self.wiring_mod == "center":
            xsum = xsum - xsum.mean(0, keepdim=True)

        wiring = F.softmax(
            xsum @ key.T / math.sqrt(self.d) * self.temp, dim=-1)
        if self.min_eff_organs > 0:
            wiring = self.enforce_min_effective(wiring, self.min_eff_organs)
        if force_mix is not None:
            fm = force_mix.unsqueeze(0) if force_mix.dim() == 1 else force_mix
            wiring = fm.to(wiring.dtype).expand(C, -1)
        elif perm is not None:
            wiring = wiring[perm]
        if self._split_child_w is not None:
            cw = self._split_child_w.to(wiring.dtype).unsqueeze(0).expand(C, -1)
            wiring = torch.cat([wiring, cw], -1)           # 正锥份额：全 n 非零

        # 计算稀疏：只精算 k 个专家；诊断仍用上面的【全 n】wiring
        topw, topi = torch.topk(wiring, self.k, dim=-1)
        if self.renorm:
            topw = topw / topw.sum(-1, keepdim=True).clamp_min(1e-9)

        out = F.silu(xc @ self.W1_base) @ self.W2_base + self.b2  # 共享基底

        need = C * self.k * d * self.h_cell * 4.0 / (1024 ** 2)
        comp = torch.zeros_like(out)
        if need > self.mem_budget_mb:
            # 逐 slot：峰值 [C,d,h_cell]，为一次性版的 1/k
            for i in range(self.k):
                E1i = self.E1.index_select(0, topi[:, i])   # [C,d,h_cell]
                E2i = self.E2.index_select(0, topi[:, i])   # [C,h_cell,d]
                hh = F.silu(torch.bmm(xc, E1i))             # [C,P,h_cell]
                comp = comp + topw[:, i][:, None, None] * torch.bmm(hh, E2i)
            self._lowmem_used = True
        else:
            E1s = self.E1[topi]                             # [C,k,d,h_cell]
            E2s = self.E2[topi]                             # [C,k,h_cell,d]
            hh = torch.matmul(xc.unsqueeze(1), E1s)         # [C,k,P,h_cell]
            oo = torch.matmul(F.silu(hh), E2s)              # [C,k,P,d]
            comp = torch.einsum("ck,ckpd->cpd", topw, oo)   # [C,P,d]
            self._lowmem_used = False
        out = out + self.gate * comp

        # ------------------------------------------------ 诊断（全 n 口径）
        wd = wiring.detach()
        self._last_mix = wd
        with torch.no_grad():
            self.mix_ema.mul_(0.99).add_(
                wd.mean(0).to(self.mix_ema.dtype) * 0.01)
            self.fitness_ema.mul_(0.99).add_(
                wd.mean(0).to(self.fitness_ema.dtype) * 0.01)
        sharp = wd.max(-1).values.mean().detach()
        eff_n = 1.0 / (wd * wd).sum(-1).mean().detach()
        mean = 1.0 / self.n
        var_max = (1.0 / self.n) * (1 - mean) ** 2 + \
                  ((self.n - 1) / self.n) * mean ** 2
        wvar = wd.var(0).mean().detach()
        self._diag = {
            **cm,
            "wiring_ent": -(wd * (wd + 1e-9).log()).sum(-1).mean().detach(),
            "wiring_variance": wvar,
            "wiring_var_norm": wvar / float(var_max),
            "eff_organs": eff_n,
            "mix_sharp": sharp,
            "gate": self.gate.detach().mean(),
            "uniq_experts": float(torch.unique(topi).numel()),
            "composed_width": float(self.k * self.h_cell),
            "lowmem_used": bool(self._lowmem_used),
            "mem_need_mb": float(need),
        }
        return self.head(out.reshape(N, d)), {**self._diag, "wiring": wd}

    # ---------------------------------------------------------------- 分裂
    @torch.no_grad()
    def expand(self, K, parents, device, child_w=0.0017, eps=0.3):
        """扩 n：忆点继承领地，E1/E2 零初始化（不继承母体知识）。"""
        n_old = self.n
        M = self.organelle_memory.detach()
        d = self.d
        NM = torch.stack([
            F.normalize(M[p] + eps * torch.randn(d, device=device), dim=0)
            for p in parents
        ])
        self.organelle_memory = nn.Parameter(torch.cat([M, NM], 0))
        self.E1 = nn.Parameter(torch.cat(
            [self.E1.data,
             torch.zeros(K, d, self.h_cell, device=device)], 0))
        self.E2 = nn.Parameter(torch.cat(
            [self.E2.data,
             torch.zeros(K, self.h_cell, d, device=device)], 0))
        self.register_buffer(
            "mix_ema",
            torch.cat([self.mix_ema,
                       torch.zeros(K, device=self.mix_ema.device)]))
        self.register_buffer(
            "fitness_ema",
            torch.cat([self.fitness_ema,
                       torch.zeros(K, device=self.fitness_ema.device)]))
        n_new = n_old + K
        cc = torch.zeros(n_new, n_new, device=device)
        if self._split_c is not None:
            cc[:n_old, :n_old] = self._split_c          # 旧块逐位不动
        self._split_c = cc
        self._split_child_w = torch.full((K,), float(child_w), device=device)
        self.n = n_new
        return n_new

    def stats(self):
        s = super().stats()
        s.update(wiring_mod=self.wiring_mod,
                 n_all=int(self.n),
                 split_n0=int(self._split_n0))
        return s
