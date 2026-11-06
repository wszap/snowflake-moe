import torch, torch.nn.functional as F
torch.manual_seed(0)
N, P, d, r, h, n = 256, 16, 128, 4, 32, 4
W1_base = torch.randn(d, h) / d**.5
W1_V    = torch.randn(r, h) / r**.5
U       = torch.randn(n, d, r) / d**.5
key     = F.normalize(torch.randn(n, d), dim=1)
x       = torch.randn(N, P, d)
logits  = x.mean(1) @ key.T / d**.5

def layer(lg, U_all):
    w   = F.softmax(lg, dim=-1)
    U_w = torch.einsum('cn,ndr->cdr', w, U_all)
    xA  = torch.einsum('npd,ndr->npr', x, U_w)
    return F.silu(x @ W1_base + xA @ W1_V)

before = layer(logits, U)
U_new  = torch.cat([U, torch.zeros(1, d, r)], 0)

print('B1  真实架构 softmax wiring，加零细胞：')
for bias in [-30.0, -10.0, 0.0]:
    b  = torch.tensor(bias, requires_grad=True)
    lg = torch.cat([logits, b.expand(N, 1)], -1)
    out = layer(lg, U_new)
    delta = (out - before).abs().max().item()
    grad  = torch.autograd.grad(out.pow(2).mean(), b)[0].abs().item()
    print(f'    bias={bias:>6}  Δ={delta:.3e}   |dL/dbias|={grad:.3e}')

b  = torch.tensor(-30.0, requires_grad=True)
lg = torch.cat([logits, b.expand(N, 1)], -1)
d30 = (layer(lg, U_new) - before).abs().max().item()
g30 = torch.autograd.grad(layer(lg, U_new).pow(2).mean(), b)[0].abs().item()
print('=' * 56)
print(f'  {"PASS" if d30 < 1e-6 else "FAIL"}  B1  负偏置出生 → 分裂无扰动')
print(f'  {"PASS" if g30 > 0 else "FAIL"}  B1b 偏置有梯度 → 连接能自己长起来')
