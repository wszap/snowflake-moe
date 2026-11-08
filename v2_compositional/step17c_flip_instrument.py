"""把 FlipperTrap.forward 逐段拆开，定位 ‖y‖ 在哪一步丢。
基已证实正交归一 → 若代数恒等式仍破，必是实现/广播细节。"""
import math, torch, torch.nn.functional as F
from flipper import FlipperTrap, _gram_schmidt

torch.manual_seed(0)
d, B, T = 256, 2, 16
x = torch.randn(B, T, d)

def n(t): return t.norm(-1).mean().item()

def instrument(ab):
    f = FlipperTrap(d)
    with torch.no_grad():
        f.alpha_net.bias.fill_(ab)
        u = f.axis_raw.view(1,1,d).expand(B,T,d)
        u = F.normalize(u + f.axis_in(x), dim=-1)
        x_par  = (x*u).sum(-1,keepdim=True)*u
        x_perp = x - x_par
        s,e = _gram_schmidt(u,
                f.seal_raw.view(1,1,d).expand(B,T,d),
                f.edge_raw.view(1,1,d).expand(B,T,d))
        gate = torch.sigmoid(f.seal_net(x_perp)/f.temp)
        trapped = gate*x_perp
        free = (1.0-gate)*x_perp
        carry = f.thickness*torch.zeros(B,d) + trapped.mean(1)
        alpha = torch.tanh(f.alpha_net(carry).unsqueeze(1))*math.pi*f.alpha_gain
        fs = (x_perp*s).sum(-1,keepdim=True)
        fe = (x_perp*e).sum(-1,keepdim=True)
        ca,sa = torch.cos(alpha), torch.sin(alpha)
        fs_r, fe_r = ca*fs - sa*fe, sa*fs + ca*fe
        rest = free - fs*s - fe*e
        rot  = fs_r*s + fe_r*e
        out_perp = rest + rot
        inject = f.carry_proj(carry).unsqueeze(1)
        y = x_par + out_perp + inject
    print(f'\n=== alpha_bias={ab}  α={alpha.mean().item():.4f}  gate={gate.mean().item():.2e} ===')
    print(f'  ‖x‖={n(x):.6f}  ‖x_par‖={n(x_par):.6f}  ‖x_perp‖={n(x_perp):.6f}')
    print(f'  ‖free‖={n(free):.6f}  ‖rest‖={n(rest):.6f}')
    print(f'  ‖fs·s+fe·e‖={n(fs*s+fe*e):.6f}   ‖rot‖={n(rot):.6f}   ← 旋转前后应相等')
    print(f'  ‖out_perp‖={n(out_perp):.6f}   ‖y‖={n(y):.6f}   ratio={n(y)/n(x):.6f}')
    print(f'  [恒等式] ‖free − (rest+fs·s+fe·e)‖ = {(free-(rest+fs*s+fe*e)).norm().item():.3e}  (应=0)')
    print(f'  [恒等式] ‖fs·s+fe·e‖ − ‖rot‖ = {n(fs*s+fe*e)-n(rot):+.6f}  (应=0)')
    print(f'  [正交]  ⟨x_par, out_perp⟩ = {(x_par*out_perp).sum(-1).abs().mean().item():.3e}  (应≈0)')
    print(f'  inject 范数 = {n(inject.expand_as(x)):.3e}  (应=0)')

instrument(0.0)
instrument(1.0)
