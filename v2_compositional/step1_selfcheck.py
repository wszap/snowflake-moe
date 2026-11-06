import torch, torch.nn.functional as F
torch.manual_seed(0)
d, h, rank, N = 128, 32, 4, 512
W1 = torch.randn(d, h) / d**.5
W2 = torch.randn(h, d) / h**.5
C  = torch.randn(d, rank) / d**.5

def cell(x, m, V):
    s = torch.sigmoid(x @ m * 4.0).unsqueeze(1)
    return F.silu(x @ W1 + (s * (x @ C)) @ V) @ W2

def layer_add(x, cells):                 # 模式A: 调制相加 = v3/v4
    mod = torch.zeros_like(x @ W1)
    for m, V in cells:
        s = torch.sigmoid(x @ m * 4.0).unsqueeze(1)
        mod = mod + (s * (x @ C)) @ V
    return F.silu(x @ W1 + mod) @ W2

def layer_mean(x, cells):                # 模式B: 独立输出平均 = v8
    return torch.stack([cell(x, m, V) for m, V in cells], 0).mean(0)

x = torch.randn(N, d)
m_p = F.normalize(torch.randn(d), dim=0)
V_p = torch.randn(rank, h) * 0.1
mk = lambda e: (F.normalize(m_p + e * torch.randn(d), dim=0), torch.zeros(rank, h))
c1, c2 = mk(0.3), mk(0.3)

d_add  = (layer_add (x, [(m_p,V_p),c1,c2]) - layer_add (x, [(m_p,V_p)])).abs().max().item()
d_mean = (layer_mean(x, [(m_p,V_p),c1,c2]) - layer_mean(x, [(m_p,V_p)])).abs().max().item()
print(f'Q2  模式A 调制相加(v3/v4) Δ = {d_add:.3e}   应≈0')
print(f'Q2b 模式B 独立平均(v8)   Δ = {d_mean:.3e}   应>0 (对照)')

V1 = torch.zeros(rank, h, requires_grad=True)
V2 = torch.zeros(rank, h, requires_grad=True)
g1 = torch.autograd.grad(cell(x, c1[0], V1).pow(2).mean(), V1)[0]
g2 = torch.autograd.grad(cell(x, c2[0], V2).pow(2).mean(), V2)[0]
cos = F.cosine_similarity(g1.flatten(), g2.flatten(), dim=0).item()
n1, n2 = g1.norm().item(), g2.norm().item()
print(f'Q1  两子细胞梯度余弦 = {cos:.4f}   <0.9 会分化')
print(f'Q1b 梯度范数 = {n1:.4f} / {n2:.4f}   >0 即无死锁')

dirv = F.normalize(torch.randn(d), dim=0)
sp = torch.sigmoid(x @ m_p * 4) > 0.5
ovs = []
for e in [0.05, 0.1, 0.3, 1.0]:
    m = F.normalize(m_p + e * dirv, dim=0)
    ov = (sp & (torch.sigmoid(x @ m * 4) > 0.5)).float().mean().item()
    ovs.append(ov)
    print(f'Q3  忆点偏移 eps={e:<5} 母子选择重叠 = {ov:.3f}')

checks = [
    ('B1 分裂无扰动(调制相加)', d_add < 1e-6),
    ('B2 模式对照(v8 会扰动)', d_mean > 1e-3),
    ('B3 子细胞梯度不对称',   cos < 0.9),
    ('B4 零初始化无死锁',     n1 > 0 and n2 > 0),
    ('B5 忆点偏移单调分化',   all(ovs[i] >= ovs[i+1] - 1e-6 for i in range(3))),
]
print('=' * 60)
for name, ok in checks:
    print(f'  {"PASS" if ok else "FAIL"}  {name}')
print(f'  {sum(o for _, o in checks)}/{len(checks)} 通过')
