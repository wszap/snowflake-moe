"""① 修好变量名 ② 直接量逐位置差，定位能量在哪一步跨位置搬运。"""
import math, torch, torch.nn.functional as F
from flipper import FlipperTrap

d, B, T = 256, 2, 16

def rg(y,x): return (y.norm()/x.norm()).item()
def rm(y,x): return (y.norm(-1).mean()/x.norm(-1).mean()).item()

print('=== ① 复刻 step17b 的抽签顺序：连建 4 个 f ===')
torch.manual_seed(0); x2 = torch.randn(B,T,d)
for i,ab in enumerate([0.0,0.5,1.0,3.0]):
    g = FlipperTrap(d)
    with torch.no_grad():
        g.alpha_net.bias.fill_(ab)
        yy, cc, dg = g(x2, carry=None)          # ★ 不再用 _ 接收
    print(f'  f#{i} ab={ab:<4} global={rg(yy,x2):.6f}  mean={rm(yy,x2):.6f}'
          f'  α={dg["alpha_mean"]:.4f}')

print('\n=== ② 同一个 f：逐位置范数差（逐位置守恒则全 0）===')
torch.manual_seed(0); x3 = torch.randn(B,T,d); f = FlipperTrap(d)
for ab in [0.0,1.0,3.0]:
    with torch.no_grad():
        f.alpha_net.bias.fill_(ab)
        y3, _, dg = f(x3, carry=None)
    nx, ny = x3.norm(-1), y3.norm(-1)           # (B,T)
    dif = ny - nx
    print(f'  ab={ab:<4} Σ‖y‖²−Σ‖x‖²={(ny.pow(2).sum()-nx.pow(2).sum()).item():+.3e}'
          f'  Σ‖y‖−Σ‖x‖={(ny.sum()-nx.sum()).item():+.3e}'
          f'  逐位置 max|Δ|={dif.abs().max().item():.3e}'
          f'  正/负个数={(dif>0).sum().item()}/{(dif<0).sum().item()}')
print('\n判读：')
print('  逐位置 max|Δ|≈0  → mean 的跳是口径数学性质，无跨位置耦合')
print('  逐位置 max|Δ|>0 且有正有负 → 确有跨位置重分配，就是翻盖在做的事')
