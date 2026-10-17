# -*- coding: utf-8 -*-
"""
Snowflake v4 —— 在「拼专家」核心不变的前提下，系统性尝试改进变体。

【核心不变】
    - 忆点 organelle_memory（384 个）
    - 细胞器低秩因子 W1_U/W1_V, W2_U/W2_V
    - 无 router：wiring = softmax(x @ key^T * temp)，自发分工
    - 核架 Scaffold（因果关联 + 筛选）
    - 忆点互相关联 key = m + c.T @ m

【新增开关】--variant 逗号分隔，可组合
    shared : 共享全秩专家 —— 补「有效秩」
             （细胞器 rank=4 × eff≈2.8 ≈ 有效秩 11，dense FFN 是全秩）
    gate   : 细胞门控 —— 4 个 cell 由「平均」改为可学习加权
    pair   : 二阶协作 —— (w@P) ⊙ (w@Q)
             数学上等于 Σ_ij w_i w_j (P_i⊙Q_j)，即器官两两协作，
             但复杂度只有 O(n·d) 而不是 O(n²·d)
    all    : 三者全开

【注意】三者都是 per-token 操作，不引入任何跨 token 信息流，
        因此不会破坏因果性（leak_check 应仍为 0）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffold_v3 import Scaffold, SafetyGate
from snowflake_B import SnowflakeB


class SharedExpert(nn.Module):
    """共享全秩专家：每个 token 都经过，负责补有效秩。

    细胞器路径的有效秩约为 rank × eff ≈ 11，而 dense FFN 是全秩。
    这一层用 d → mult*d → d 的全秩 MLP 补上这个缺口。
    """

    def __init__(self, d, mult=4):
        super().__init__()
        self.fc1 = nn.Linear(d, d * mult)
        self.fc2 = nn.Linear(d * mult, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SnowflakeLMV4(nn.Module):
    def __init__(self, vocab_size, d=128, n=384, L=4, h=32, n_cells=4,
                 seq_len=128, rank=4, temp=5.0, n_head=4,
                 scaffold_top_k=32, select_mode="soft",
                 variant="base", shared_mult=4, seed=0,
                 risk_threshold=0.5):
        super().__init__()
        self.d = d
        self.L = L
        self.n_cells = n_cells
        self.seq_len = seq_len
        self.select_mode = select_mode

        vs = set(v.strip() for v in variant.split(",") if v.strip())
        if "all" in vs:
            vs = {"shared", "gate", "pair"}
        self.variants = vs

        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(seq_len, d)
        self.in_proj = nn.Linear(d, d, bias=False)

        self.scaffolds = nn.ModuleList([
            Scaffold(d, n_head=n_head, top_k=scaffold_top_k,
                     select_mode=select_mode)
            for _ in range(L)
        ])

        self.layers = nn.ModuleList([
            nn.ModuleList([
                SnowflakeB(d, n_organelles=n, h=h, rank=rank,
                           seq_len=seq_len, chunk_size=1,
                           temp=temp, hinge_mode="loss",
                           use_hinge_low=False, log_sharp=True,
                           min_eff_organs=0,
                           seed=seed * 1000 + li * 10 + ci)
                for ci in range(n_cells)
            ]) for li in range(L)
        ])

        # ---- shared：共享全秩专家 ----
        self.shared = None
        if "shared" in vs:
            self.shared = nn.ModuleList(
                [SharedExpert(d, shared_mult) for _ in range(L)])

        # ---- gate：每个 cell 的可学习权重 ----
        self.cell_logits = None
        if "gate" in vs:
            self.cell_logits = nn.ParameterList([
                nn.Parameter(torch.zeros(n_cells)) for _ in range(L)
            ])

        # ---- pair：二阶协作 (w@P) ⊙ (w@Q) ----
        self.pair_P = None
        self.pair_Q = None
        self.pair_proj = None
        if "pair" in vs:
            self.pair_P = nn.ParameterList([
                nn.Parameter(torch.randn(n, d) * (1.0 / math.sqrt(d)))
                for _ in range(L)
            ])
            self.pair_Q = nn.ParameterList([
                nn.Parameter(torch.randn(n, d) * (1.0 / math.sqrt(d)))
                for _ in range(L)
            ])
            self.pair_proj = nn.ModuleList(
                [nn.Linear(d, d, bias=False) for _ in range(L)])
            # 零初始化：训练开始时 pair 项贡献为 0，退化为 v3 行为
            for li in range(L):
                nn.init.zeros_(self.pair_proj[li].weight)

        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.final_norm = nn.LayerNorm(d)
        self.safety = SafetyGate(d, risk_threshold=risk_threshold)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, tokens, perm=None, force_mix=None):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.in_proj(self.embed(tokens) + self.pos(pos))
        diag_last = {}
        wiring_last = None

        for li, (sc, layer) in enumerate(zip(self.scaffolds, self.layers)):
            # ① 核架：因果关联
            ctx = sc(x)
            x = x + ctx

            # ② 细胞器：忆点接收上下文，自发分工
            outs = []
            for cell in layer:
                y, info = cell(x.reshape(B * T, self.d),
                               perm=perm, force_mix=force_mix)
                outs.append(y.reshape(B, T, self.d))
                diag_last = info
                if "wiring" in info:
                    wiring_last = info["wiring"]

            # ③ cell 组合：平均 或 可学习门控
            if self.cell_logits is not None:
                g = F.softmax(self.cell_logits[li], dim=-1)
                delta = sum(gi * o for gi, o in zip(g, outs))
            else:
                delta = sum(outs) / len(outs)

            # ④ 共享全秩专家：补有效秩
            if self.shared is not None:
                delta = delta + self.shared[li](x)

            # ⑤ 二阶协作：(w@P) ⊙ (w@Q)
            if self.pair_P is not None and wiring_last is not None:
                w = wiring_last.reshape(B * T, -1)
                w = w.to(self.pair_P[li].dtype)
                second = (w @ self.pair_P[li]) * (w @ self.pair_Q[li])
                delta = delta + self.pair_proj[li](second).view(B, T, self.d)

            x = x + delta
            x = self.norms[li](x)

        x, safety_info = self.safety(self.final_norm(x), wiring=wiring_last)
        diag_last["safety"] = safety_info
        diag_last["variants"] = sorted(self.variants)
        return self.head(x), diag_last

    def heal_sum(self):
        s = None
        for layer in self.layers:
            for cell in layer:
                if cell.heal_term is not None:
                    s = cell.heal_term if s is None else s + cell.heal_term
        return s
