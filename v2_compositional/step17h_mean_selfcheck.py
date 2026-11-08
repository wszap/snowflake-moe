import torch
from flipper import FlipperTrap
torch.manual_seed(0)
d,B,T = 256,2,16
x = torch.randn(B,T,d)
print(f'x.shape={tuple(x.shape)}  ‖x‖F={x.norm().item():.6f} '
      f'x.norm(-1).shape={tuple(x.norm(-1).shape)}  x.norm(-1).mean()={x.norm(-1).mean().item():.6f}')
for i,ab in enumerate([0.0,0.5,1.0,3.0]):
    f = FlipperTrap(d)
    with torch.no_grad():
        f.alpha_net.bias.fill_(ab)
        y,_,dg = f(x,carry=None)
    a, b = y.norm(-1), x.norm(-1)          # 均 (B,T)
    print(f'  f#{i} ab={ab:<4} α={dg["alpha_mean"]:.4f}')
    print(f'     y.norm(-1).shape={tuple(a.shape)}  a.mean={a.mean().item():.6f}  '
          f'b.mean={b.mean().item():.6f}  a.mean/b.mean={(a.mean()/b.mean()).item():.6f}')
    print(f'     a.sum−b.sum={(a.sum()-b.sum()).item():+.3e}  '
          f'‖y‖F/‖x‖F={(y.norm()/x.norm()).item():.6f}  '
          f'a[:2,:4]={a[:2,:4].flatten().tolist()[:4]}')
print('\n若 a.mean/b.mean ≠ 1 但 a.sum−b.sum≈0 → 脚本里 rm 的 x 被遮蔽，口径 bug，弃用 mean')
