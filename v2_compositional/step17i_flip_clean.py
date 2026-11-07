# -*- coding: utf-8 -*-
"""翻盖收口：修正 norm 口径后一次性重跑。
规矩：全局 Frobenius 用 t.norm()；逐位置 L2 必须 t.norm(dim=-1)。
      禁止 t.norm(-1)（-1 是 p 不是 dim → 0 维标量 → 索引即 IndexError）。
      任何切片前先 assert ndim/shape。
"""
import ast, os, torch
from flipper import FlipperTrap

torch.manual_seed(0)
d, B, T = 256, 2, 16
x = torch.randn(B, T, d)
assert x.dim() == 3

FRO = lambda t: t.norm()            # 0 维 Frobenius
POS = lambda t: t.norm(dim=-1)      # (B,T) 逐位置 L2   ★ 关键字，不能用位置参数

def report(tag, y, dg, show_slice=False):
    assert y.shape == x.shape, f'y.shape={tuple(y.shape)}'
    ny, nx = POS(y), POS(x)
    assert ny.dim() == 2 and nx.dim() == 2, '若 0 维 → norm(-1) 又写错了'
    rF = (FRO(y) / FRO(x)).item()
    rP = (ny.mean() / nx.mean()).item()
    dif = ny - nx
    print(f'  {tag:<24} F={rF:.6f}  pos={rP:.6f}  '
          f'Σ差={dif.sum().item():+.3e}  max|Δ|={dif.abs().max().item():.3e}  '
          f'α={float(dg["alpha_mean"]):.4f}  gate={float(dg["gate_mean"]):.2e}')
    if show_slice:
        assert ny.shape[0] >= 2 and ny.shape[1] >= 4, '切片越界'
        print(f'     ny[:2,:4] = {[round(v,3) for v in ny[:2,:4].flatten().tolist()]}')
    return rF, rP

print('=' * 78)
print('[1] 同一个 f，α 真转起来：两种口径都应严格 1')
f = FlipperTrap(d)
for ab in [0.0, 0.5, 1.0, 3.0]:
    with torch.no_grad():
        f.alpha_net.bias.fill_(ab)
        y, _, dg = f(x, carry=None)
    report(f'alpha_bias={ab}', y, dg, show_slice=(ab == 0.0))

print('\n[2] gate 扫描（Frobenius 口径，重定 gate_max）')
print(f'  {"seal_bias":>10}{"gate_mean":>12}{"‖y‖F/‖x‖F":>14}{"判定":>10}')
clean_max = 0.0
for sb in [-8, -4, -2, -1, 0, 1, 2, 4, 8]:
    g = FlipperTrap(d)
    with torch.no_grad():
        g.seal_net.bias.fill_(float(sb))
        y, _, dg = g(x, carry=None)
    rF = (FRO(y) / FRO(x)).item()
    gm = float(dg['gate_mean'])
    ok = rF > 0.98
    if ok:
        clean_max = max(clean_max, gm)
    print(f'  {sb:>10}{gm:>12.4f}{rF:>14.4f}{("无损" if ok else "有损"):>10}')
print(f'  → gate_max ≈ {clean_max:.2e}  （覆盖旧 flip_gate.log 的 ratio 列）')

print('\n[3] α 解耦复核')
if not os.path.exists('flipper_decoupled.py'):
    s = open('flipper.py').read()
    old = 'alpha = self.alpha_net(carry)'
    assert old in s, '锚点没找到：把 flipper.py 里 alpha 那一行贴给我'
    s = s.replace(old, 'alpha = self.alpha_net(x_perp.mean(dim=1))', 1)
    s = s.replace('class FlipperTrap(', 'class FlipperTrapDecoupled(', 1)
    open('flipper_decoupled.py', 'w').write(s)
    ast.parse(s)
    print('  已生成 flipper_decoupled.py（只改 1 处驱动源，保留 .unsqueeze(1) 与注释）')
from flipper_decoupled import FlipperTrapDecoupled

a, b = FlipperTrap(d), FlipperTrapDecoupled(d)
with torch.no_grad():
    y0, _, _ = a(x, carry=None)
    y1, _, _ = b(x, carry=None)
    b.alpha_net.weight.normal_(0, 0.1)
    b.alpha_net.bias.zero_()
    y2, _, d2 = b(x, carry=None)

checks = [
    ('默认严格恒等', (y1 - x).abs().max().item() < 1e-5),
    ('默认与原版逐位一致', (y1 - y0).abs().max().item() < 1e-5),
    ('α 真转 > 0.05', abs(float(d2['alpha_mean'])) > 0.05),
    ('严格保范数 (Frobenius)', abs((FRO(y2) / FRO(x)).item() - 1.0) < 1e-4),
    ('gate 在无损窗口', float(d2['gate_mean']) < 1e-5),
    ('逐位置也守恒', abs((POS(y2).mean() / POS(x).mean()).item() - 1.0) < 1e-4),
]
for n, ok in checks:
    print(f'  {"PASS" if ok else "FAIL"}  {n}')
print(f'  {sum(o for _, o in checks)}/{len(checks)} 通过')
print('\n完成：[1] 两种口径都=1 → mean 矛盾消解；[2] gate_max 重定；[3] 解耦复核')
