import torch
from flipper import FlipperTrap
torch.manual_seed(0)
d,B,T = 256,2,16
x = torch.randn(B,T,d)
def probe(f, ab=None):
    if ab is not None:
        with torch.no_grad(): f.alpha_net.bias.fill_(ab)
    with torch.no_grad(): y,_,dg = f(x, carry=None)
    return (y.norm(-1).mean()/x.norm(-1).mean()).item(), dg
print('=== ⑤ 真旋转（直接给 α bias，绕过 carry 依赖）===')
for b in [0.0,0.5,1.0,3.0]:
    r,dg = probe(FlipperTrap(d), b)
    print(f'  alpha_bias={b:<4} ‖y‖/‖x‖={r:.6f}  α={dg["alpha_mean"]:.4f}  gate={dg["gate_mean"]:.2e}')
print('\n=== ⑥ gate 扫描（找无损窗口）===')
print(f'{"seal_bias":>10}{"gate":>12}{"‖y‖/‖x‖":>12}{"判定":>10}')
for sb in [-8,-4,-2,-1,0,1,2,4,8]:
    f=FlipperTrap(d)
    with torch.no_grad(): f.seal_net.bias.fill_(float(sb))
    r,dg = probe(f)
    print(f'{sb:>10}{dg["gate_mean"]:>12.4f}{r:>12.4f}{("无损" if r>0.98 else "改终点"):>10}')
r,dg = probe(FlipperTrap(d), 1.0)
print('\n判据: α>0.05 且 ‖y‖/‖x‖≈1 且 gate<1e-5 → 纯旋转保范数成立')
print(f'  实测 α={dg["alpha_mean"]:.4f}  ratio={r:.6f}  gate={dg["gate_mean"]:.2e}')
