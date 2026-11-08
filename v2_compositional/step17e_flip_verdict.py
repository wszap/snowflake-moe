"""裁决 step17b 的 0.6685：同一个 f / 同一个 x / 两种范数口径 / 逐个 f 查基。"""
import math, torch, torch.nn.functional as F
from flipper import FlipperTrap, _gram_schmidt

d,B,T = 256,2,16

def ratio_global(y,x): return (y.norm()/x.norm()).item()
def ratio_mean(y,x):   return (y.norm(-1).mean()/x.norm(-1).mean()).item()

def base_quality(f):
    torch.manual_seed(0); xx = torch.randn(B,T,d)      # 与下面同一次抽签
    u = F.normalize(f.axis_raw.view(1,1,d).expand(B,T,d),dim=-1)
    s,e = _gram_schmidt(u, f.seal_raw.view(1,1,d).expand(B,T,d),
                           f.edge_raw.view(1,1,d).expand(B,T,d))
    s0,e0,u0=s[0,0],e[0,0],u[0,0]
    return (s0.norm().item(),e0.norm().item(),
            (s0*e0).sum().item(),(s0*u0).sum().item())

print('=== ① 同一个 f，扫 alpha_bias，两种口径并排 ===')
torch.manual_seed(0)
x = torch.randn(B,T,d); f = FlipperTrap(d)
for ab in [0.0,0.5,1.0,3.0]:
    with torch.no_grad():
        f.alpha_net.bias.fill_(ab)
        y,_,dg = f(x,carry=None)
    print(f'  ab={ab:<4} α={dg["alpha_mean"]:.4f} gate={dg["gate_mean"]:.2e}'
          f'  global={ratio_global(y,x):.6f}  mean={ratio_mean(y,x):.6f}')

print('\n=== ② 复刻 step17b：连续新建 4 个 f，看基有没有退化 ===')
torch.manual_seed(0); _ = torch.randn(B,T,d)            # 消耗同一次 x
for i,ab in enumerate([0.0,0.5,1.0,3.0]):
    g = FlipperTrap(d)
    with torch.no_grad():
        g.alpha_net.bias.fill_(ab); yy,_,_ = g(_,carry=None)
    ns,ne,se,su = base_quality(g)
    print(f'  f#{i} ab={ab:<4} ‖s‖={ns:.4f} ‖e‖={ne:.4f} ⟨s,e⟩={se:+.4f} ⟨s,u⟩={su:+.4f}'
          f'  global={ratio_global(yy,_):.6f}  mean={ratio_mean(yy,_):.6f}')

print('\n判读：')
print('  global=mean=1 全部 → 0.6685 纯属脚本污染，保范数成立（采信 step17d）')
print('  global=1 但 mean<1 → 存在跨位置重分配，是真发现，需继续查')
print('  某个 f 的 ⟨s,e⟩≠0 或 ‖s‖≠1 → Gram-Schmidt 在该初始化下退化，0.6685 有了实现层解释')
