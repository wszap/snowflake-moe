# -*- coding: utf-8 -*-
"""同参数量 inert 对照：复现参数/梯度/裁剪系数，旁路旋转功能。"""
import os, torch, torch.nn as nn
import train_split
from flipper_decoupled import FlipperTrapDecoupled

class DummyFlip(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.inner = FlipperTrapDecoupled(d)      # 参数量逐位一致
    def forward(self, x, carry=None):
        y, c, diag = self.inner(x, carry)
        return x + (y - y.detach()), c, diag      # 前向严格=x；反向梯度与真档一致

assert os.environ.get('FLIP_MODE') == 'dummy'
_orig = train_split.build
def build_patched(a, vocab, n_start, device):
    m = _orig(a, vocab, n_start, device)
    cpu_st = torch.get_rng_state()
    cuda_st = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    dev = next(m.parameters()).device
    m.flippers = nn.ModuleList([DummyFlip(a.d).to(dev) for _ in range(a.L)])
    torch.set_rng_state(cpu_st)
    if cuda_st is not None: torch.cuda.set_rng_state_all(cuda_st)
    print(f'[dummy_ab] L={a.L} dev={dev}', flush=True)
    return m
train_split.build = build_patched
train_split.main()
