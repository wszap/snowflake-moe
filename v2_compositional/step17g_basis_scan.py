import torch, torch.nn.functional as F
from flipper import FlipperTrap, _gram_schmidt

torch.manual_seed(0)
d, B, T = 256, 2, 16
x = torch.randn(B, T, d)
rg = lambda y: (y.norm()/x.norm()).item()
rm = lambda y: (y.norm(-1).mean()/x.norm(-1).mean()).item()

print('=== 逐个 f：基正交性 + 两种口径 + 逐位置差 ===')
for i, ab in enumerate([0.0, 0.5, 1.0, 3.0]):
    g = FlipperTrap(d)
    with torch.no_grad():
        g.alpha_net.bias.fill_(ab)
        y, _, dg = g(x, carry=None)
        u = F.normalize(g.axis_raw.view(1,1,d).expand(B,T,d) + g.axis_in(x), dim=-1)
        s, e = _gram_schmidt(u,
                g.seal_raw.view(1,1,d).expand(B,T,d),
                g.edge_raw.view(1,1,d).expand(B,T,d))
    s0, e0, u0 = s[0,0], e[0,0], u[0,0]
    nx, ny = x.norm(-1), y.norm(-1)
    dif = ny - nx
    print(f'\n  f#{i} ab={ab:<4} α={dg["alpha_mean"]:.4f} gate={dg["gate_mean"]:.2e}')
    print(f'    基   ‖s‖={s0.norm().item():.6f} ‖e‖={e0.norm().item():.6f} '
          f'⟨s,e⟩={(s0*e0).sum().item():+.6f} '
          f'⟨s,u⟩={(s0*u0).sum().item():+.6f} ⟨e,u⟩={(e0*u0).sum().item():+.6f}')
    print(f'    范数 global={rg(y):.6f} mean={rm(y):.6f} '
          f'Σ²差={(ny.pow(2).sum()-nx.pow(2).sum()).item():+.3e} '
          f'Σ差={(ny.sum()-nx.sum()).item():+.3e} '
          f'max|Δ|={dif.abs().max().item():.3e} '
          f'正/负={(dif>0).sum().item()}/{(dif<0).sum().item()}')
print('\n判读：')
print('  f#1~3 基仍正交且 max|Δ|≈0 → mean 纯属跨实例口径伪影，弃用 mean')
print('  f#1~3 ⟨s,e⟩或‖s‖明显偏离 → _gram_schmidt 的 +1e-8 退化，是可修的实现 bug')
print('  max|Δ|大且正负都有、Σ²差≈0 → 那几档确有跨位置重分配，单独命名，不泛化')
