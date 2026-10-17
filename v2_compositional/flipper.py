"""
翻转式下水道塞 (Flip-Trap) —— 四维机制
================================================

物理装置对应：
    一维  两个支点          → 轴 u（可移动，但永远在对面、过中心）
    二维  密封圈那一圈      → 软门槛，分选"细碎的"和"打结的"
    三维  厚度              → 兜住扣下来的团（capacity）
    四维  翻转角 α          → 状态，跨位置/跨 chunk 传递

工作流：
    ① 轴 u      : 沿轴分量 x_par 一动不动（支点稳住，主信息永不丢失）
    ② 密封圈    : 按"纠缠度"软门槛分选 → trapped(打结) / free(细碎)
    ③ 厚度      : trapped 累积进 carry（兜住，不是丢弃）
    ④ 翻转角 α  : free 绕轴等距旋转；α 与 carry 一起传到下游
                  （B 方案：团被送到下游集中处理，不是丢弃）

由结构保证、非靠正则化的性质：
    · 平行分量严格不变（恒等通道）
    · 翻转是等距旋转（不放大不衰减 → 数值不跑飞）
    · 两个支点恒在对面（轴 ±u 结构性对称）
    · α 可跨位置传递 → 自带记忆，补 chunk 间断流
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gram_schmidt(u, s_raw, e_raw):
    """把 s_raw / e_raw 正交化到垂直于 u，且 s ⊥ e。
    返回正交单位向量 s, e（均与 u 正交）。"""
    # s ⊥ u
    s = s_raw - (s_raw * u).sum(-1, keepdim=True) * u
    s = F.normalize(s + 1e-8, dim=-1)
    # e ⊥ u 且 e ⊥ s
    e = e_raw - (e_raw * u).sum(-1, keepdim=True) * u
    e = e - (e * s).sum(-1, keepdim=True) * s
    e = F.normalize(e + 1e-8, dim=-1)
    return s, e


class FlipperTrap(nn.Module):
    """翻转式下水道塞。

    输入 : x (B, T, d) —— 本步新产生的信息（如细胞器输出 delta）
    输出 : y (B, T, d) —— 分选 + 翻转后送往下流的信息
    状态 : carry (B, d) —— 兜住的团，跨位置累积传递（B 方案）
    """

    def __init__(self, d, temp=0.5, thickness=0.9, init_gate_bias=-8.0,
                 alpha_gain=0.1, shared_axis=True):
        super().__init__()
        self.d = d
        self.temp = temp
        self.thickness = thickness          # ③ 厚度：兜住容量（carry 衰减率）
        self.alpha_gain = alpha_gain

        # ① 轴 u：两个支点（基础轴，可随输入移动）
        self.axis_raw = nn.Parameter(torch.randn(d))
        if shared_axis:
            self.axis_in = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.axis_in.weight)   # 零初始化→起步时轴不随输入变
        else:
            self.axis_in = None

        # ② 密封圈：二维圈（赤道环），软门槛
        self.seal_raw = nn.Parameter(torch.randn(d))
        self.edge_raw = nn.Parameter(torch.randn(d))
        # 纠缠度检测：细碎信息照常流走，打结的被扣下
        self.seal_net = nn.Linear(d, 1)
        nn.init.zeros_(self.seal_net.weight)
        nn.init.constant_(self.seal_net.bias, init_gate_bias)  # 起步几乎不扣留

        # ③ 厚度：carry 的注入投影（把兜住的团送回下游）
        self.carry_proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.carry_proj.weight)   # 零初始化→起步退化为无 flipper

        # ④ 翻转角 α：由 trapped 的量驱动（攒够了才翻）
        self.alpha_net = nn.Linear(d, 1)
        nn.init.zeros_(self.alpha_net.weight)
        nn.init.zeros_(self.alpha_net.bias)

    def forward(self, x, carry=None):
        B, T, d = x.shape
        dev = x.device

        # ---- ① 轴 u：两个支点永远在对面（±u 过中心）----
        u = self.axis_raw.to(dev).view(1, 1, d).expand(B, T, d)
        if self.axis_in is not None:
            u = u + self.axis_in(x)          # 支点可移动（随输入）
        u = F.normalize(u, dim=-1)

        # 分解：沿轴 / 垂直轴
        x_par = (x * u).sum(-1, keepdim=True) * u     # 支点稳住：一动不动
        x_perp = x - x_par                            # 交给翻转片

        # ---- ② 密封圈：软门槛分选 ----
        s, e = _gram_schmidt(
            u,
            self.seal_raw.to(dev).view(1, 1, d).expand(B, T, d),
            self.edge_raw.to(dev).view(1, 1, d).expand(B, T, d),
        )
        tangle = self.seal_net(x_perp)                        # (B,T,1) 纠缠度
        gate = torch.sigmoid(tangle / self.temp)              # 软门槛（橡胶有弹性）

        trapped = gate * x_perp        # 打结的团 → 扣下
        free = (1.0 - gate) * x_perp   # 细碎的 → 照常流走

        # ---- ③ 厚度：兜住团，累积进 carry ----
        if carry is None:
            carry = torch.zeros(B, d, device=dev, dtype=x.dtype)
        carry = self.thickness * carry + trapped.mean(dim=1)   # 沿位置累积

        # ---- ④ 翻转角 α：攒够了才翻，free 绕轴等距旋转 ----
        alpha = self.alpha_net(carry).unsqueeze(1)             # (B,1,1)
        alpha = torch.tanh(alpha) * math.pi * self.alpha_gain

        # 在 (s, e) 平面内等距旋转（严格保范数 → 不跑飞）
        fs = (x_perp * s).sum(-1, keepdim=True)   # 密封圈法向分量
        fe = (x_perp * e).sum(-1, keepdim=True)
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        fs_r = ca * fs - sa * fe
        fe_r = sa * fs + ca * fe
        rest = free - fs * s - fe * e             # 其余垂直分量不变
        out_perp = rest + fs_r * s + fe_r * e

        # ⑤ B 方案：兜住的团送到下游（注入回去），不是丢弃
        inject = self.carry_proj(carry).unsqueeze(1)

        y = x_par + out_perp + inject
        diag = {
            "gate_mean": float(gate.mean().detach()),
            "carry_norm": float(carry.norm(dim=-1).mean().detach()),
            "alpha_mean": float(alpha.mean().detach()),
            "par_frac": float((x_par.norm(dim=-1).mean()
                               / (x.norm(dim=-1).mean() + 1e-8)).detach()),
        }
        return y, carry, diag
