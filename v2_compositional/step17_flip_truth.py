import torch
from flipper import FlipperTrap

torch.manual_seed(0)
d, B, T = 256, 2, 16
x = torch.randn(B, T, d)

def run(name, f):
    with torch.no_grad():
        y, carry, diag = f(x, carry=None)
    r = (y.norm(-1).mean() / x.norm(-1).mean()).item()
    dl = (y - x).abs().max().item()
    print(f'{name:<24} ‖y‖/‖x‖={r:.6f}  Δmax={dl:.3e}  '
          f'gate={diag["gate_mean"]:.2e}  α={diag["alpha_mean"]:.3e}')
    return r, dl, diag

print('=' * 78)
# ① 原样：四个零初始化 + bias=-8 → 恒等起步
f0 = FlipperTrap(d);              r0, d0, g0 = run('① 默认(恒等起步)', f0)
# ② 只开旋转：α 变大，gate 仍≈0 → 应严格保范数
f1 = FlipperTrap(d)
with torch.no_grad(): f1.alpha_net.weight.normal_(0, 0.5)
r1, d1, g1 = run('② 只旋转(gate≈0)', f1)
# ③ 只开扣留：gate→1，inject=0 → 范数应下降（信息被兜住）
f2 = FlipperTrap(d)
with torch.no_grad(): f2.seal_net.bias.fill_(8.0)
r2, d2, g2 = run('③ 只扣留(inject=0)', f2)
# ④ 扣留 + 尽量补偿：carry_proj 给单位阵，仍只能补回「位置平均」
f3 = FlipperTrap(d)
with torch.no_grad():
    f3.seal_net.bias.fill_(8.0)
    f3.carry_proj.weight.copy_(torch.eye(d))
r3, d3, g3 = run('④ 扣留+位置均值补偿', f3)

print('\n' + '=' * 78)
checks = [
 ('① 恒等起步 y≈x（零初始化 + gate≈0）', d0 < 1e-4),
 ('① gate 起步 < 1e-5', g0['gate_mean'] < 1e-5),
 ('② 纯旋转严格保范数（Givens）', abs(r1 - 1.0) < 1e-4 and g1['gate_mean'] < 1e-5),
 ('③ 扣留阶段信息被兜住（‖y‖<‖x‖）', r2 < r1 - 0.01 and g2['gate_mean'] > 0.5),
 ('④ 位置均值补偿无法完全还原（有损重分配）', r3 < 1.0 - 1e-3),
]
for name, ok in checks: print(f'  {"PASS" if ok else "FAIL"}  {name}')
print(f'  {sum(o for _,o in checks)}/{len(checks)} 通过')
print('  含义：旋转=加速(不改信息量)；gate+carry=跨位置重分配(终点需 A/B 实测)')
