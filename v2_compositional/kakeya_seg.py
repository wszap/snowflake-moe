"""
挂谷翻转塞 · 换挂载位置：从「逐 token 空间变换」改为「沿序列段传递」
=====================================================================

★ 为什么要换（v6 的实测结论）
--------------------------------------------------
v6 把翻转塞挂在每层每个 token 的 delta 上（逐位置空间变换）。结果：

    K=0  基线        val 1.7288
    K=8  α=0 起步    val 1.7286   α 绝对均值 0.0132  ← 模型压根没启用
    K=8  α=0.1 强制  val 1.7422   α 被压回 0.0750   ← 用了，反而更差

在那个位置上，翻转塞变成「又一个逐 token 变换层」，跟 dense FFN 抢同一份
工作且抢不过；模型主动想关掉它。它真正该解决的是另一件事：
      【位置与位置之间信息流是断的】
      —— 难缠的、纠缠成一团的东西没人攒着往下游带。

★ 新挂载点：沿序列段推进
--------------------------------------------------
序列 T 切成 S 段，维护「翻转片上的存货」h：

    段 s 读： 用【上一段末尾】留下的 h（只含段 1..s-1 ⇒ 严格因果）
    段 s 翻：  h ← 绕轴旋转（Givens）  ← 两个支点稳住，保范
    段 s 攒：  h ← h + 注入门 × 本段摘要 g_s（把团扣在片上）
    段 s 写：  out_s = seg_s + 输出门 × h

★ 重要修正：用 Givens 旋转，不用 Householder 插值
--------------------------------------------------
之前用的 y = x − 2α(x·u)u，分解后沿轴分量被乘 (1−2α)：

    α=0   → ×1     恒等      ✅
    α=1   → ×−1    保范      ✅
    α=0.5 → ×0     清零      ❌
    α=0.66→ ×−0.33 缩短      ❌

【只有 α=0 或 α=1 才保范，中间任何值都在收缩】。跨位置传递时 carry 会被
逐段衰减到 0 —— 对状态传递是致命的。

而且物理建模错了：翻转片是【绕两个支点连成的轴】转，在 d 维里是
【在一个平面内旋转】= Givens 旋转；不是 Householder 反射（那是镜像）。

    Givens 旋转任意角度都保范 —— 转 30° 和转 180° 一样。
    这才是刚体，才配得上「两个支点稳住，永远不掉」。

零初始化保证：θ=0 → 旋转为 I；门=0 → 不攒不送 ⇒ 起步严格不存在。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def givens_rotate(x, u, v, theta):
    """绕 (u,v) 张成的平面旋转 theta（Givens 旋转）

    严格保范：只在平面内转，垂直分量一动不动。
    u, v 需已正交归一；theta=0 时严格恒等。

    x     : (..., d)
    u, v  : (..., d)
    theta : 标量或 (...,)
    """
    a = (x * u).sum(-1)
    b = (x * v).sum(-1)
    c = torch.cos(theta)
    s = torch.sin(theta)
    return (x
            + (a * (c - 1.0) - b * s).unsqueeze(-1) * u
            + (a * s + b * (c - 1.0)).unsqueeze(-1) * v)


class SegmentKakeyaFlipper(nn.Module):
    """沿序列段推进的挂谷翻转塞（Givens 版）。"""

    def __init__(self, d, n_anchors=384, K=8, n_seg=8, temp=1.0):
        super().__init__()
        self.d = d
        self.K = K
        self.n_seg = n_seg
        self.temp = temp
        self.n_anchors = n_anchors

        # ① 旋转角（零初始化 → 起步不转，严格恒等）
        self.theta = nn.Parameter(torch.zeros(K))

        # ② 注入门：什么该被扣在翻转片上（零初始化 → 什么都不扣）
        self.inj_gate = nn.Parameter(torch.zeros(1))

        # ③ 输出门：攒着的东西送多少到下游（零初始化 → 不送）
        self.out_gate = nn.Parameter(torch.zeros(1))

        # ④ 衰减门：密封圈可以「留住」一点（零初始化 → 不衰减）
        self.decay_gate = nn.Parameter(torch.zeros(1))

        self.anchors = nn.Parameter(torch.randn(n_anchors, d) * 0.1)

    def forward(self, x, anchors=None):
        B, T, d = x.shape
        S = self.n_seg
        if T % S != 0:
            return x                      # 兜底：不整除就不启用，绝不崩

        seg_len = T // S
        A = anchors if anchors is not None else self.anchors
        if A is None:
            return x
        A_n = F.normalize(A, dim=-1)                        # (n, d)

        th = self.theta                                     # (K,)
        inj = torch.tanh(self.inj_gate)
        out = torch.tanh(self.out_gate)
        keep = torch.sigmoid(self.decay_gate)               # ∈(0,1)

        h = x.new_zeros(B, d)                               # 翻转片上的存货
        outs = []

        for s in range(S):
            seg = x[:, s * seg_len:(s + 1) * seg_len, :]

            # ---- 读：把【上一段留下的存货】交给本段（严格因果）----
            outs.append(seg + out * h.unsqueeze(1))

            # ---- 攒：本段摘要 = 这一段的「团」----
            g = seg.mean(1)

            # ---- 翻：检索 2K 个轴，配对成正交平面，依次旋转 ----
            scores = torch.matmul(g, A_n.transpose(0, 1))    # (B, n)
            _, idx = torch.topk(scores / self.temp, 2 * self.K, dim=-1)
            P = F.normalize(A_n[idx], dim=-1)                # (B, 2K, d)

            for k in range(self.K):
                u = P[:, 2 * k, :]
                v = P[:, 2 * k + 1, :]
                # Gram-Schmidt：让 v 垂直于 u（保证是真正的平面旋转）
                v = v - (v * u).sum(-1, keepdim=True) * u
                nv = v.norm(dim=-1)
                # 退化保护：u,v 近乎平行时 v 会趋近 0，此时旋转会退化成
                # 「半边转」而破坏保范 ⇒ 直接跳过（等价于 θ=0，恒等）
                degen = nv < 1e-6
                v = v / nv.clamp(min=1e-12).unsqueeze(-1)
                th_eff = torch.where(
                    degen, torch.zeros_like(nv), nv.new_full(nv.shape, th[k]))
                h = givens_rotate(h, u, v, th_eff)

            # ---- 留：密封圈（可选衰减，默认不衰减）----
            h = h * keep

            # ---- 扣：本段的团存到片上 ----
            h = h + inj * g

        return torch.cat(outs, dim=1)

    def extra_repr(self):
        return f"d={self.d}, K={self.K}, n_seg={self.n_seg}"


class SegKakeyaLMV3(nn.Module):
    """在 v3 每层接入【沿段推进】的挂谷翻转塞（作用在整条残差流上）。"""

    def __init__(self, kakeya_k=8, n_seg=8, share_memory=True, **kw):
        super().__init__()
        from scaffold_v3 import SnowflakeLMV3

        self.base = SnowflakeLMV3(**kw)
        d = self.base.d
        L = self.base.L
        self.kakeya_k = kakeya_k
        self.n_seg = n_seg

        n_anchors = kw.get("n", 384)
        self.flippers = nn.ModuleList([
            SegmentKakeyaFlipper(d, n_anchors=n_anchors,
                                 K=max(kakeya_k, 1), n_seg=n_seg)
            for _ in range(L)
        ])
        self._share_memory = share_memory

    def _memory_anchors(self, li):
        cell = self.base.layers[li][0]
        return getattr(cell, "organelle_memory", None)

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

            # ★ 沿段推进的翻转塞（跨位置信息通道）
            if self.kakeya_k > 0:
                x = self.flippers[li](
                    x, self._memory_anchors(li) if self._share_memory else None)

            outs, wiring_last = [], None
            for cell in layer:
                y, info = cell(x.reshape(B * T, base.d),
                               perm=perm, force_mix=force_mix)
                outs.append(y.reshape(B, T, base.d))
                diag_last = info
                if "wiring" in info:
                    wiring_last = info["wiring"]

            delta = sum(outs) / len(outs)
            x = x + delta
            x = base.norms[li](x)

        x, safety_info = base.safety(base.final_norm(x), wiring=wiring_last)
        diag_last["safety"] = safety_info
        diag_last["kakeya_k"] = self.kakeya_k
        diag_last["kakeya_where"] = "segment"
        return base.head(x), diag_last

    def heal_sum(self):
        return self.base.heal_sum()
