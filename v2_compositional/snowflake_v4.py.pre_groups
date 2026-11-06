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
    clone  : 复制-联结（用户核心机制，2026-10-17）
             ① 需求  细胞器从当前问题提出 demand
             ② 选择  细胞器【自己】从库里挑想复制的标本 spec_w
             ③ 复制  忆点执行 —— 加权复制标本能力 + 可学习变体
             ④ 联结  忆点把【提出者】与【复制体】绑成临时团队
             ⑤ 求解  团队一起解决同一个问题
             组合空间：n → n×n = 384×384 = 147,456
             可配 --clone-temp 控制「复制几种标本」（默认 2.0）
    all    : shared + gate + pair（不含 clone，clone 请单独指定以免归因混淆）

【注意】三者都是 per-token 操作，不引入任何跨 token 信息流，
        因此不会破坏因果性（leak_check 应仍为 0）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffold_v3 import Scaffold, SafetyGate
from snowflake_B import SnowflakeB
from flipper import FlipperTrap


class SnowflakeBClone(SnowflakeB):
    """『复制-联结』型拼专家 —— 细胞器自己组队

    严格按用户 2026-10-17 描述的机制实现：

      ① 需求  活跃细胞器从【当前问题】提出需求 demand
      ② 选择  细胞器【自己】从细胞器库里挑想复制的标本（不是被指派）
      ③ 复制  忆点执行 —— 加权复制标本能力 + 可学习变体
      ④ 联结  忆点把【提出者】和【复制体】联结成临时团队
      ⑤ 求解  团队一起解决【同一个】问题

    ★ 为什么这是「无限组合」
        提出者可以是 384 个中的任意一个（wiring 决定）
        复制的标本也可以是 384 个中的任意一个（spec_w 决定）
        ⇒ 组合数 384 × 384 = 147,456，而不是 384

    ★ 与标准 MoE 的本质区别
        标准 MoE：router 挑 k 个专家，各干各的，加权求和
        本机制：细胞器主动【复制】库里的标本当队友，
                由【忆点】把两者联结成一个团队，团队共同求解
    """

    def __init__(self, *args, clone_temp=2.0, **kwargs):
        seed = kwargs.get("seed", 0)
        super().__init__(*args, **kwargs)
        d, n = self.d, self.n
        self.clone_temp = clone_temp

        g = torch.Generator().manual_seed(seed * 7919 + 13)
        # ① 需求探针：从当前状态提取「我缺什么」
        #    用标准 1/sqrt(d) 初始化，使 demand 量级与 x 相当
        #    （若尺度过小，spec_w 会退化成近似均匀 ⇒ 失去「选择性」）
        self.demand_probe = nn.Parameter(
            torch.randn(d, d, generator=g) * (1.0 / math.sqrt(d)))
        # ③ 复制变体：让复制体不是标本的照搬（1.0 = 纯复制起步）
        self.clone_scale = nn.Parameter(torch.ones(n, 1, 1))
        # ④ 联结强度：忆点决定两个细胞器绑多紧（0 → sigmoid=0.5）
        self.link_gate = nn.Parameter(torch.zeros(1))
        self.link_probe = nn.Parameter(
            torch.randn(n, 1, generator=g) * 0.01)

    def forward(self, x, perm=None, force_mix=None):
        N, d = x.shape
        c, cm = self.connect()
        key = self.organelle_memory + c.T @ self.organelle_memory

        xc = self._chunk_view(x)
        C, P, _ = xc.shape
        xsum = xc.mean(1)

        # ---- 原有：哪些细胞器被叫醒 ----
        wiring = F.softmax(xsum @ key.T / math.sqrt(self.d) * self.temp, dim=-1)
        if self.min_eff_organs > 0:
            wiring = self.enforce_min_effective(wiring, self.min_eff_organs)
        if force_mix is not None:
            fm = force_mix.unsqueeze(0) if force_mix.dim() == 1 else force_mix
            wiring = fm.to(wiring.dtype).expand(C, -1)
        elif perm is not None:
            wiring = wiring[perm]

        # ---- ★① 需求：细胞器从当前问题提出需求 ----
        demand = torch.matmul(xsum, self.demand_probe)          # [C, d]

        # ---- ★② 选择：细胞器自己从【库】里挑想复制的标本 ----
        lib_logit = torch.matmul(demand, key.T) / math.sqrt(self.d) * self.clone_temp
        spec_w = F.softmax(lib_logit, dim=-1)                   # [C, n]

        # ---- ★③ 复制：忆点执行，加权复制标本能力 ----
        U_lib = self.W1_U * self.clone_scale                    # [n,d,r]
        U_clone = torch.einsum('cn,ndr->cdr', spec_w, U_lib)    # [C,d,r]

        # ---- ★④ 联结：忆点把提出者与复制体绑成团队 ----
        link = torch.sigmoid(
            self.link_gate + torch.matmul(spec_w, self.link_probe))     # [C,1]
        U_self = torch.matmul(
            wiring, self.W1_U.view(self.n, -1)).view(-1, self.d, self.rank)
        U_team = U_self + link.view(C, 1, 1) * U_clone

        # ---- ★⑤ 团队一起解决同一个问题 ----
        xA = torch.bmm(xc, U_team)
        hh = F.silu(torch.matmul(xc, self.W1_base)
                    + torch.matmul(xA, self.W1_V))

        # W2 用同一批队友（团队保持一致）
        U2_lib = self.W2_U * self.clone_scale
        U2_clone = torch.einsum('cn,nhr->chr', spec_w, U2_lib)
        U2_self = torch.matmul(
            wiring, self.W2_U.view(self.n, -1)).view(-1, self.h, self.rank)
        U2_team = U2_self + link.view(C, 1, 1) * U2_clone

        hA = torch.bmm(hh, U2_team)
        out = (torch.matmul(hh, self.W2_base)
               + torch.matmul(hA, self.W2_V))

        # ---------------- 诊断 ----------------
        wd = wiring.detach()
        sd = spec_w.detach()
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

        # 复制体诊断
        eff_spec = 1.0 / (sd * sd).sum(-1).mean().detach()      # 复制了几种标本
        with torch.no_grad():
            ns = U_self.detach().norm()
            nc = U_clone.detach().norm()
            gain = float(link.detach().mean()) * float(nc / (ns + 1e-9))

        self._diag = {**cm,
                      "wiring_ent": -(wd * (wd + 1e-9).log()).sum(-1).mean().detach(),
                      "wiring_variance": wvar,
                      "wiring_var_norm": wvar / float(var_max),
                      "eff_organs": eff_n,
                      "mix_sharp": sharp,
                      "clone_eff": eff_spec,
                      "clone_link": float(link.detach().mean()),
                      "clone_team_gain": gain}
        return self.head(out.reshape(N, d)), {**self._diag, "wiring": wd}


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
                 risk_threshold=0.5, clone_temp=2.0):
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

        def make_cell(li, ci):
            kw = dict(d=d, n_organelles=n, h=h, rank=rank,
                      seq_len=seq_len, chunk_size=1,
                      temp=temp, hinge_mode="loss",
                      use_hinge_low=False, log_sharp=True,
                      min_eff_organs=0,
                      seed=seed * 1000 + li * 10 + ci)
            if "clone" in vs:
                return SnowflakeBClone(clone_temp=clone_temp, **kw)
            return SnowflakeB(**kw)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                make_cell(li, ci)
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

        # ---- flipper：翻转式下水道塞（四维机制）----
        self.flippers = None
        if "flipper" in vs:
            self.flippers = nn.ModuleList([FlipperTrap(d) for _ in range(L)])

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
        carry = None

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

            # ⑥ 翻转塞：分选打结的团 → 兜住 → 送下游（B 方案）
            if self.flippers is not None:
                delta, carry, fdiag = self.flippers[li](delta, carry=carry)
                diag_last["flipper"] = fdiag

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
