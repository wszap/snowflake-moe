"""
挂谷翻转塞 (Kakeya Flipper) —— 多轴 Householder 反射
====================================================

物理装置 → 算法（承接 flipper.py 的四维设计）
    一维  两个支点      → 翻转轴 u（从忆点库检索，永远在对面 ±u）
    二维  密封圈        → 软门槛（此处简化：由检索分数隐式承担）
    三维  厚度          → 兜住能力（由 alpha 的幅度承担）
    四维  翻转动作      → 翻转角 α ∈ [-1, 1]

数学内核
    一次完全翻转 = 一个 Householder 反射
        y = x - 2 (x·u) u    =    (I - 2 u uᵀ) x
    这是等距、自逆（H² = I）的正交变换。

    Cartan–Dieudonné 定理：d 维空间中任意正交变换，
    可分解为至多 d 个反射的复合。
    ⇒ K = d 个翻转塞串联，可表示【任意正交变换】= 满秩 d。

★ 最关键的一点（务必理解，否则会误判）
--------------------------------------------------
    正交变换【不改变秩】。
    若 K 个轴对所有 token 都相同（固定轴 M），
        rank(X Mᵀ) = rank(X)      ← 秩一点没变！
    秩的提升【只能】来自「轴随输入变化」：
        rank([x_1 M_1ᵀ ; x_2 M_2ᵀ ; ...]) > rank([x_i])
    不同 token 检索到不同轴 → 被送到不同方向 → 整体覆盖维度扩大。

    这正是挂谷集合的算法对应：
        挂谷集合：测度(体积)可趋近 0，但维数必须是 n（满）
        翻转塞  ：参数极少（K 个 α + d 个门），却能覆盖所有方向
    「参数少」与「方向全」并不矛盾 —— 数学上明确成立。

零初始化保证（加了不会变差）
    α 初始化为 0   → y = x - 0·(...) = x     严格恒等
    对角门初始化 0 → 乘 (1 + tanh(0)) = 1    严格恒等
    ⇒ 起步时本模块完全退化为直通，与 v3 数值一致。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. 核心算子
# ============================================================================
def householder_apply(x, u, alpha):
    """一次翻转：y = x - 2α (x·u) u

    x     : (..., d)
    u     : (..., d)  已归一化的翻转轴（两个支点，永远在对面）
    alpha : (...,) 或标量，翻转强度
            0 = 不翻（恒等）   1 = 完全翻面（标准 Householder）
    """
    proj = (x * u).sum(-1, keepdim=True) * u
    return x - 2.0 * alpha * proj


class KakeyaFlipper(nn.Module):
    """多轴翻转塞：检索 K 个轴 → 依次翻转 → 对角门补缩放。

    参数量极少：K（翻转角）+ d（对角门）= 约 136 个（d=128, K=8）
    轴本身复用忆点库（organelle_memory），不新增方向参数。
    """

    def __init__(self, d, n_anchors=384, K=8, use_diag=True,
                 share_anchors=False, temp=1.0):
        super().__init__()
        self.d = d
        self.K = K
        self.n_anchors = n_anchors
        self.use_diag = use_diag
        self.temp = temp

        # ④ 翻转角：零初始化 → 严格恒等（★ 保证加了不会变差）
        self.alpha = nn.Parameter(torch.zeros(K))

        # 备选：若不共享忆点，自建锚点库（从 v3 的忆点统计量初始化更稳）
        if share_anchors:
            self.anchors = None          # 由外部传入（忆点库）
        else:
            self.anchors = nn.Parameter(torch.randn(n_anchors, d) * 0.1)

        # 对角门：正交变换不能缩放，补一个廉价的对角 Σ（SVD 的 Σ 部分）
        if use_diag:
            self.diag_gate = nn.Parameter(torch.zeros(d))
        else:
            self.register_buffer("diag_gate", torch.zeros(d))

    def forward(self, x, anchors=None):
        """
        x       : (B, T, d)  待翻转的信息（如细胞器 delta）
        anchors : (n, d) 锚点库（忆点方向）；None 时用自带的
        返回     : (B, T, d) 翻转后结果
        """
        B, T, d = x.shape
        A = anchors if anchors is not None else self.anchors
        if A is None:
            return x

        A_n = F.normalize(A, dim=-1)                            # (n, d)

        # ---- ① 检索：每个 token 自己从忆点库挑 K 个翻转轴 ----
        #        不同 token → 不同轴 → 覆盖不同方向（★ 秩提升的来源）
        scores = torch.matmul(x, A_n.transpose(0, 1))            # (B,T,n)
        _, idx = torch.topk(scores / self.temp, self.K, dim=-1)  # (B,T,K)
        U = F.normalize(A_n[idx], dim=-1)                        # (B,T,K,d)

        # ---- ④ 依次翻转（K 个 Householder 反射串联）----
        a = torch.tanh(self.alpha)                               # (K,) ∈ (-1,1)
        y = x
        for k in range(self.K):
            y = householder_apply(y, U[..., k, :], a[k])

        # ---- 对角门：补正交变换缺失的缩放能力 ----
        if self.use_diag:
            y = y * (1.0 + torch.tanh(self.diag_gate))

        return y

    def extra_repr(self):
        return f"d={self.d}, K={self.K}, n_anchors={self.n_anchors}"


# ============================================================================
# 2. 接入 v3 模型（挂谷版）
# ============================================================================
class KakeyaLMV3(nn.Module):
    """在 SnowflakeLMV3 的每层插入挂谷翻转塞。

    插在哪：细胞器输出 delta 上（细胞器替换的是 FFN，delta 的有效秩
            被 rank=4 锁死，翻转塞负责把它拉满）。
    """

    def __init__(self, kakeya_k=8, kakeya_on_delta=True,
                 kakeya_use_diag=True, share_memory=True, **kw):
        super().__init__()
        from scaffold_v3 import SnowflakeLMV3

        self.base = SnowflakeLMV3(**kw)
        d = self.base.d
        L = self.base.L
        self.kakeya_k = kakeya_k
        self.kakeya_on_delta = kakeya_on_delta

        n_anchors = kw.get("n", 384)
        self.flippers = nn.ModuleList([
            KakeyaFlipper(d, n_anchors=n_anchors, K=kakeya_k,
                          use_diag=kakeya_use_diag,
                          share_anchors=share_memory)
            for _ in range(L)
        ])

    def _memory_anchors(self, li):
        """取该层第一个细胞器的忆点库作为锚点（复用，不新增参数）"""
        cell = self.base.layers[li][0]
        mem = getattr(cell, "organelle_memory", None)
        return mem

    def forward(self, tokens, perm=None, force_mix=None):
        base = self.base
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)

        x = base.in_proj(base.embed(tokens) + base.pos(pos))
        diag_last = {}
        wiring_last = None

        for li, (sc, layer) in enumerate(zip(base.scaffolds, base.layers)):
            ctx = sc(x)
            x = x + ctx

            outs, wiring_last = [], None
            for cell in layer:
                y, info = cell(x.reshape(B * T, base.d),
                               perm=perm, force_mix=force_mix)
                outs.append(y.reshape(B, T, base.d))
                diag_last = info
                if "wiring" in info:
                    wiring_last = info["wiring"]

            delta = sum(outs) / len(outs)

            # ★ 挂谷翻转塞：把被 rank=4 锁死的 delta 拉满秩
            #   kakeya_k=0 时完全跳过（基线，与 v3 数值一致）
            if self.kakeya_on_delta and self.kakeya_k > 0:
                delta = self.flippers[li](delta, self._memory_anchors(li))

            x = x + delta
            x = base.norms[li](x)

        x, safety_info = base.safety(base.final_norm(x), wiring=wiring_last)
        diag_last["safety"] = safety_info
        diag_last["kakeya_k"] = self.kakeya_k
        return base.head(x), diag_last

    def heal_sum(self):
        return self.base.heal_sum()
