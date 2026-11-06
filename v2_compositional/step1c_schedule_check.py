import torch, torch.nn.functional as F
torch.manual_seed(0)
N, P, d, r, h, n = 256, 16, 128, 4, 32, 4
W1_base = torch.randn(d, h) / d**.5
W1_V    = torch.randn(r, h) / r**.5
W2      = torch.randn(h, d) / h**.5          # 共享投影回 d 维
U       = torch.randn(n, d, r) / d**.5
key     = F.normalize(torch.randn(n, d), dim=1)
x       = torch.randn(N, P, d)
target  = torch.randn_like(x)                # [N,P,d]
logits  = x.mean(1) @ key.T / d**.5          # [N,n]

def layer(lg, U_all):
    w   = F.softmax(lg, dim=-1)              # [N, n_all]
    U_w = torch.einsum('cn,ndr->cdr', w, U_all)
    xA  = torch.einsum('npd,ndr->npr', x, U_w)
    return F.silu(x @ W1_base + xA @ W1_V) @ W2    # [N,P,d]

before = layer(logits, U)

print('Part1  出生偏置扫描：找「扰动可接受 + 新知识有梯度」的窗口')
print(f'{"bias":>7}{"Δ输出":>14}{"w_child":>12}{"|dL/dU_new|":>16}')
window = []
for bias in [-8., -5., -3., -2., -1., 0.]:
    child = torch.zeros(1, d, r, requires_grad=True)
    U_all = torch.cat([U, child], 0)
    lg = torch.cat([logits, torch.full((N, 1), bias)], -1)
    out = layer(lg, U_all)
    delta = (out - before).abs().max().item()
    wch = F.softmax(lg, -1)[:, -1].mean().item()
    gu = torch.autograd.grad(out.pow(2).mean(), child)[0].norm().item()
    print(f'{bias:>7.1f}{delta:>14.3e}{wch:>12.4f}{gu:>16.3e}')
    if delta < 0.05 and gu > 1e-4:
        window.append(bias)

print()
print('Part2  显式日程：bias 从 -3 线性升到 0，只看新细胞有没有学到东西')
U_fixed = U.clone()
child = torch.zeros(1, d, r, requires_grad=True)
opt = torch.optim.SGD([child], lr=0.05)
norms = []
for step in range(201):
    b = -3.0 + 3.0 * min(step / 100, 1.0)      # 前 100 步接线
    lg = torch.cat([logits, torch.full((N, 1), b)], -1)
    U_all = torch.cat([U_fixed, child], 0)
    loss = (layer(lg, U_all) - target).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    norms.append(child.norm().item())
    if step in (0, 50, 100, 200):
        print(f'  step {step:>3}  bias={b:>5.2f}  loss={loss.item():.4f}  ||U_new||={norms[-1]:.4f}')

print('=' * 62)
print(f'  {"PASS" if window else "FAIL"}  Part1 存在可用出生窗口 bias={window}')
print(f'  {"PASS" if norms[-1] > norms[0] + 1e-3 else "FAIL"}  Part2 ||U_new|| 从 0 增长 ({norms[0]:.4f} -> {norms[-1]:.4f})')
