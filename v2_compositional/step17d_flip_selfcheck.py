"""先让脚本证明自己的尺子准，再判根因。
★ 同一个 f 实例、同一个 x、全局 Frobenius 范数（无口径歧义）。"""
import math, torch, torch.nn.functional as F
from flipper import FlipperTrap, _gram_schmidt

torch.manual_seed(0)
d, B, T = 256, 2, 16
x = torch.randn(B, T, d)
f = FlipperTrap(d)                         # ★ 只建一次
N = lambda t: t.norm().item()             # ★ 全局范数
print(f'x.shape={tuple(x.shape)}  ‖x‖²={N(x)**2:.6f}')

def check(ab):
    with torch.no_grad():
        f.alpha_net.bias.fill_(ab)
        u = F.normalize(f.axis_raw.view(1,1,d).expand(B,T,d) + f.axis_in(x), dim=-1)
        x_par  = (x*u).sum(-1,keepdim=True)*u
        x_perp = x - x_par
        s,e = _gram_schmidt(u, f.seal_raw.view(1,1,d).expand(B,T,d),
                               f.edge_raw.view(1,1,d).expand(B,T,d))
        gate = torch.sigmoid(f.seal_net(x_perp)/f.temp)
        free = (1.0-gate)*x_perp
        carry = f.thickness*torch.zeros(B,d) + (gate*x_perp).mean(1)
        alpha = torch.tanh(f.alpha_net(carry).unsqueeze(1))*math.pi*f.alpha_gain
        fs = (x_perp*s).sum(-1,keepdim=True); fe = (x_perp*e).sum(-1,keepdim=True)
        ca,sa = torch.cos(alpha), torch.sin(alpha)
        rest  = free - fs*s - fe*e
        rot   = (ca*fs-sa*fe)*s + (sa*fs+ca*fe)*e
        proj  = fs*s + fe*e
        out_perp = rest + rot
        inject = f.carry_proj(carry).unsqueeze(1)
        y_man  = x_par + out_perp + inject
        y_true, _, dg = f(x, carry=None)      # ★ 真 forward，同 f 同 x
    print(f'\n=== alpha_bias={ab}  α={dg["alpha_mean"]:.4f}  gate={dg["gate_mean"]:.2e} ===')
    print(f'  [尺子] ‖x‖²={N(x)**2:.6f}  ‖x_par‖²+‖x_perp‖²={N(x_par)**2+N(x_perp)**2:.6f}'
          f'  差={N(x)**2-N(x_par)**2-N(x_perp)**2:+.3e}  (应≈0)')
    print(f'  [旋转] ‖proj‖²={N(proj)**2:.6f}  ‖rot‖²={N(rot)**2:.6f}'
          f'  差={N(proj)**2-N(rot)**2:+.3e}  (应≈0)')
    print(f'  [正交] ⟨x_par,out_perp⟩={(x_par*out_perp).sum().item():+.3e}  (应≈0)')
    print(f'  [手工] ratio_man = {N(y_man)/N(x):.6f}')
    print(f'  [真身] ratio_true= {N(y_true)/N(x):.6f}')
    print(f'  [差]   ‖y_true−y_man‖ = {N(y_true-y_man):.3e}  (应=0；非0 → forward 还有未读到的行)')

check(0.0)
check(1.0)
