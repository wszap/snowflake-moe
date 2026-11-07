# -*- coding: utf-8 -*-
"""翻盖最后一验：只收 [3]。
规矩：Frobenius 用 t.norm()；逐位置用 t.norm(dim=-1)；禁用位置参数 norm(-1)。
      判据不靠抽签：重设 seed + 确定性 bias；索引前 assert shape。
"""
import math, torch
import torch.nn.functional as F
from flipper import FlipperTrap
from flipper_decoupled import FlipperTrapDecoupled

torch.manual_seed(0)
d, B, T = 256, 2, 16
x     = torch.randn(B, T, d)
x_alt = torch.randn(B, T, d)
FRO = lambda t: t.norm()
POS = lambda t: t.norm(dim=-1)

# ★ 重设 seed，避开 [1][2] 消耗 RNG 后新建实例的不确定性
torch.manual_seed(17)
a = FlipperTrap(d)
b = FlipperTrapDecoupled(d)

with torch.no_grad():
    y0, _, _ = a(x, carry=None)
    y1, _, _ = b(x, carry=None)          # ★ 必须在改参之前取：零初始化恒等
    b.alpha_net.bias.fill_(1.0)          # ★ 确定性：α≈0.239，不靠抽签
    b.alpha_net.weight.normal_(0, 0.01)  # 小扰动：让 α 随输入变
    y2, _, d2 = b(x, carry=None)

def alpha_of(mod, xx):
    """与 flipper.py ④ 段逐行一致；返回 (B,1,1)。"""
    assert xx.dim() == 3 and tuple(xx.shape) == (B, T, d)
    u = F.normalize(mod.axis_raw.view(1,1,d).expand(B,T,d) + mod.axis_in(xx), dim=-1)
    x_perp = xx - (xx*u).sum(-1, keepdim=True)*u
    src = x_perp.mean(dim=1)
    assert src.dim() == 2
    return torch.tanh(mod.alpha_net(src).unsqueeze(1)) * math.pi * mod.alpha_gain

al1 = alpha_of(b, x)
al2 = alpha_of(b, x_alt)
ny, nx = POS(y2), POS(x)
assert ny.dim() == 2 and tuple(ny.shape) == (B, T), '若 0 维 → norm(-1) 又写错了'

a_amp = al1.abs().mean().item()          # ★ 先 abs 再 mean，不抵消
a_dep = (al1 - al2).abs().max().item()   # ★ 解耦价值：α 由输入驱动
rF    = (FRO(y2)/FRO(x)).item()
rP    = (ny.mean()/nx.mean()).item()
gate  = float(d2['gate_mean'])

print(f'  α(x) abs.mean      = {a_amp:.4f}')
print(f'  α 随输入 max|Δ|    = {a_dep:.3e}')
print(f'  F={rF:.6f}  pos={rP:.6f}  gate={gate:.2e}')
print(f'  diag.alpha_mean    = {float(d2["alpha_mean"]):.4f}  (有符号 mean，仅参考，不作判据)')

checks = [
 ('默认严格恒等',                (y1-x).abs().max().item() < 1e-5),
 ('默认与原版逐位一致',          (y1-y0).abs().max().item() < 1e-5),
 ('α 真转起来 (abs.mean>0.05)', a_amp > 0.05),
 ('严格保范数 (Frobenius)',     abs(rF-1.0) < 1e-4),
 ('gate 在严格无损窗口 <3e-4',  gate < 3e-4),
 ('逐位置也守恒',                abs(rP-1.0) < 1e-4),
 ('★ α 由输入驱动（解耦价值）', a_dep > 1e-4),
]
print()
for n, ok in checks:
    print(f'  {"PASS" if ok else "FAIL"}  {n}')
print(f'  {sum(o for _,o in checks)}/{len(checks)} 通过')
