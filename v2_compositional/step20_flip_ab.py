# -*- coding: utf-8 -*-
"""翻盖 A/B 入口：FLIP_MODE=off|decoupled，零改动现有源码。
只做一件事：build() 之后把 model.flippers 换成解耦版，并保护全局 RNG。
"""
import sys, os, torch, torch.nn as nn
import train_split
from flipper_decoupled import FlipperTrapDecoupled

mode = os.environ.get('FLIP_MODE', 'off')
assert mode in ('off', 'decoupled'), mode
_orig_build = train_split.build

def build_patched(a, vocab, n_start, device):
    m = _orig_build(a, vocab, n_start, device)
    if mode == 'decoupled':
        # ★ 保存 RNG：新增模块不得消耗抽签，否则两档不同
        cpu_st = torch.get_rng_state()
        cuda_st = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        dev = next(m.parameters()).device
        m.flippers = nn.ModuleList(
            [FlipperTrapDecoupled(a.d).to(dev) for _ in range(a.L)])
        torch.set_rng_state(cpu_st)
        if cuda_st is not None: torch.cuda.set_rng_state_all(cuda_st)
        print(f'[flip_ab] FLIP_MODE=decoupled  L={a.L}  dev={dev}', flush=True)
    else:
        print('[flip_ab] FLIP_MODE=off', flush=True)
    return m

train_split.build = build_patched
train_split.main()
