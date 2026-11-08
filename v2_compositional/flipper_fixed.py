import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from flipper import _gram_schmidt

class FlipperFixed(nn.Module):
    """零可学习基 + 单标量增益：默认严格恒等，旋转严格保范数，无 gate/carry。"""
    def __init__(self, d, alpha_gain=0.1):
        super().__init__()
        self.register_buffer("u_raw", torch.randn(d))
        self.register_buffer("s_raw", torch.randn(d))
        self.register_buffer("e_raw", torch.randn(d))
        self.alpha_gain = alpha_gain
        self.gain = nn.Parameter(torch.zeros(1))   # 唯一可学参数；0 → α=0 → y=x
    def forward(self, x, carry=None):
        assert x.dim() == 3
        B, T, d = x.shape
        u = F.normalize(self.u_raw.view(1, 1, d), dim=-1)
        x_par  = (x * u).sum(dim=-1, keepdim=True) * u
        x_perp = x - x_par
        s, e = _gram_schmidt(u, self.s_raw.view(1, 1, d),
                                self.e_raw.view(1, 1, d))
        src  = x_perp.mean(dim=1)                              # (B,d)
        proj = (src * self.u_raw.view(1, d)).sum(dim=-1) / math.sqrt(d)
        alpha = (torch.tanh(self.gain * proj) * math.pi * self.alpha_gain).view(B, 1, 1)
        fs = (x_perp * s).sum(dim=-1, keepdim=True)
        fe = (x_perp * e).sum(dim=-1, keepdim=True)
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        rest = x_perp - fs * s - fe * e
        rot  = (ca * fs - sa * fe) * s + (sa * fs + ca * fe) * e
        y = x_par + rest + rot                                 # x_par⊥x_perp, rest⊥rot
        diag = {"alpha_mean": float(alpha.detach().abs().mean()), "gate_mean": 0.0}
        return y, x.new_zeros(B, d), diag
