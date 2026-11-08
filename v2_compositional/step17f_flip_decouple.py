"""α 解耦原型：只在副本上手工重算，不改 flipper.py。
carry  = 原版：α←carry←gate，默认 gate≈1e-7 → α≈0（恒等）
direct = 解耦：α←x_perp.mean(1)，不经过 gate → 应真转起来且严格保范数
"""
import math, torch, torch.nn.functional as F
from flipper import FlipperTrap, _gram_schmidt

torch.manual_seed(0)
d, B, T = 256, 2, 16
x = torch.randn(B, T, d)
f = FlipperTrap(d)
with torch.no_grad():
    f.alpha_net.weight.normal_(0, 0.1)   # ★ 让 α 真正依赖输入
    f.alpha_net.bias.zero_()

def manual(src_mode):
    u = F.normalize(f.axis_raw.view(1,1,d).expand(B,T,d) + f.axis_in(x), dim=-1)
    x_par  = (x*u).sum(-1,keepdim=True)*u
    x_perp = x - x_par
    s, e = _gram_schmidt(u,
            f.seal_raw.view(1,1,d).expand(B,T,d),
            f.edge_raw.view(1,1,d).expand(B,T,d))
    gate   = torch.sigmoid(f.seal_net(x_perp)/f.temp)
    free   = (1.0-gate)*x_perp
    carry  = f.thickness*torch.zeros(B,d) + (gate*x_perp).mean(1)
    src    = carry if src_mode == 'carry' else x_perp.mean(1)   # ★ 唯一差别
    alpha  = torch.tanh(f.alpha_net(src).unsqueeze(1))*math.pi*f.alpha_gain
    fs = (x_perp*s).sum(-1,keepdim=True); fe = (x_perp*e).sum(-1,keepdim=True)
    ca, sa = torch.cos(alpha), torch.sin(alpha)
    rest = free - fs*s - fe*e
    out_perp = rest + (ca*fs-sa*fe)*s + (sa*fs+ca*fe)*e
    y = x_par + out_perp + f.carry_proj(carry).unsqueeze(1)
    return y, alpha, gate

rg = lambda y: (y.norm()/x.norm()).item()
rm = lambda y: (y.norm(-1).mean()/x.norm(-1).mean()).item()

y_c, a_c, g_c = manual('carry')
with torch.no_grad():
    y_true, _, dg = f(x, carry=None)
print(f'  carry  : α={a_c.abs().mean().item():.4f} gate={g_c.mean().item():.2e} '
      f'global={rg(y_c):.6f} mean={rm(y_c):.6f}')
print(f'           ‖y_man−y_true‖={(y_c-y_true).norm().item():.3e}  (应≈0，手工路径没抄错)')

y_d, a_d, g_d = manual('direct')
print(f'  direct : α={a_d.abs().mean().item():.4f} gate={g_d.mean().item():.2e} '
      f'global={rg(y_d):.6f} mean={rm(y_d):.6f}')
print('\n判据: direct α>0.05 且 global≈1 且 gate<1e-5')
print('      → α 可不经过 gate 起效，「纯旋转、不扣留、严格保范数」的加速器成立')
