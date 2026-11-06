import torch, torch.nn as nn, torch.nn.functional as F
from snowflake_B import SnowflakeB
from organelle_registry import OrganelleRegistry
from organelle_groups import SoftGroups

torch.manual_seed(0)
d, h, r, n = 256, 128, 4, 384
m = SnowflakeB(d, n_organelles=n, h=h, rank=r, seq_len=128, chunk_size=1, temp=5.0, seed=0)
x = torch.randn(128, d)
xc = m._chunk_view(x); C = xc.shape[0]
xsum = xc.mean(1).detach()                       # ★ 切断图，允许多次 backward

with torch.no_grad():
    c0, _ = m.connect()
    key0 = m.organelle_memory + c0.T @ m.organelle_memory
    w0 = F.softmax(xsum @ key0.T / d**0.5 * m.temp, dim=-1).detach()   # ★
    out0, _ = m(x)

with torch.no_grad():                            # 加一行忆点：5 个按 n 索引的张量
    nm = F.normalize(m.organelle_memory[0] + 0.3 * torch.randn(d), dim=0).unsqueeze(0)
    m.organelle_memory = nn.Parameter(torch.cat([m.organelle_memory.data, nm], 0))
    m.W1_U = nn.Parameter(torch.cat([m.W1_U.data, torch.zeros(1, d, r)], 0))
    m.W2_U = nn.Parameter(torch.cat([m.W2_U.data, torch.zeros(1, h, r)], 0))
    m.register_buffer('mix_ema', torch.cat([m.mix_ema, torch.zeros(1)]))
    m.register_buffer('fitness_ema', torch.cat([m.fitness_ema, torch.zeros(1)]))
    m.n = n + 1

# ---- Q0 全局耦合代价 ----
with torch.no_grad():
    c1, _ = m.connect()
    key1 = m.organelle_memory + c1.T @ m.organelle_memory
    dk = (key1[:n] - key0).abs().max().item()
print(f'Q0  c {tuple(c0.shape)} -> {tuple(c1.shape)}   旧 key 扰动 = {dk:.3e}')

# ---- Q1/Q2 真类 force_mix 出生份额 ----
print(f'\n{"w_child":>10}{"Δ输出":>14}{"|g_W1_U|":>14}{"|g_W2_U|":>14}')
rows = []
for wch in [0.0, 0.0017, 0.0123, 0.0327]:
    fm = torch.cat([w0, torch.full((C, 1), wch)], -1)   # 前 n 项保持 w0，不缩放
    m.zero_grad()
    out, _ = m(x, force_mix=fm)
    delta = (out - out0).abs().max().item()
    out.pow(2).mean().backward()
    g1 = m.W1_U.grad[-1].norm().item(); g2 = m.W2_U.grad[-1].norm().item()
    rows.append((wch, delta, g1, g2))
    print(f'{wch:>10.4f}{delta:>14.3e}{g1:>14.3e}{g2:>14.3e}')

# ---- Q3 主动接线：自身忆点命中率驱动，知识从零长 ----
reg = OrganelleRegistry(d, n + 1)
child = torch.zeros(1, d, r, requires_grad=True)
opt = torch.optim.SGD([child], lr=0.05)
target = torch.randn(C, d, r)
norms = []
for step in range(120):
    wc = reg.wire_in(n, xsum, m.organelle_memory, eta=0.02, cap=0.05)
    fm = torch.cat([w0, torch.full((C, 1), wc)], -1)
    U = torch.cat([m.W1_U.data[:n], child], 0)
    U1 = torch.matmul(fm, U.view(n + 1, -1)).view(C, d, r)
    loss = (U1 - target).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    norms.append(child.norm().item())
    if step in (0, 60, 119):
        print(f'Q3  step {step:>3}  w_child={wc:.5f}  ||U_new||={norms[-1]:.4f}')

# ---- Q4 软群：仍是拼 ----
g = SoftGroups(d, n + 1, G=8)
w, w_g = g.wiring(xsum, m.organelle_memory.detach())
health = g.health(m.organelle_memory.detach())
print(f'Q4  min(w)={w.min().item():.3e}  每行非零={(w>0).sum(-1).float().mean().item():.0f}/{n+1}'
      f'  熵比={health["entropy_ratio"]:.4f}  P互余弦={health["P_max_cos"]:.4f}')

# ---- 五条判据，全部从数据算 ----
checks = [
 ('Q0 connect 自动适配', c1.shape == (n + 1, n + 1)),
 ('Q1 零初始化分裂严格无扰动', all(dl < 1e-5 for _, dl, _, _ in rows)),
 ('Q2 w_child>0 时新知识有梯度', any(dl < 1e-5 and (g1 > 1e-6 or g2 > 1e-6)
                                    for wc, dl, g1, g2 in rows if wc > 0)),
 ('Q3 主动接线推动知识从零增长', norms[-1] > norms[0] + 1e-3),
 ('Q4 软群仍是拼(全非零+未塌缩)', w.min().item() > 0 and health['P_max_cos'] < 0.99),
]
print('\n' + '=' * 62)
for name, ok in checks: print(f'  {"PASS" if ok else "FAIL"}  {name}')
print(f'  {sum(o for _, o in checks)}/{len(checks)} 通过')
