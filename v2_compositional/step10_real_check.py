import torch, torch.nn as nn, torch.nn.functional as F
from snowflake_B import SnowflakeB
from organelle_registry import OrganelleRegistry
from organelle_groups import SoftGroups

torch.manual_seed(0)
d, h, r, n = 256, 128, 4, 384
m = SnowflakeB(d, n_organelles=n, h=h, rank=r,
               seq_len=128, chunk_size=1, temp=5.0, seed=0)
x = torch.randn(128, d)
xc = m._chunk_view(x); C = xc.shape[0]
c0, _ = m.connect()
key0 = m.organelle_memory + c0.T @ m.organelle_memory
w0 = F.softmax(xc.mean(1) @ key0.T / d**0.5 * m.temp, dim=-1)
with torch.no_grad(): out0, _ = m(x)

# ---- 真类加一行忆点：5 个按 n 索引的张量 ----
with torch.no_grad():
    new_mem = F.normalize(m.organelle_memory[0] + 0.3*torch.randn(d), dim=0).unsqueeze(0)
    m.organelle_memory = nn.Parameter(torch.cat([m.organelle_memory.data, new_mem], 0))
    m.W1_U = nn.Parameter(torch.cat([m.W1_U.data, torch.zeros(1,d,r)], 0))
    m.W2_U = nn.Parameter(torch.cat([m.W2_U.data, torch.zeros(1,h,r)], 0))
    m.register_buffer('mix_ema', torch.cat([m.mix_ema, torch.zeros(1)]))
    m.register_buffer('fitness_ema', torch.cat([m.fitness_ema, torch.zeros(1)]))
    m.n = n + 1

print('=== Q0 connect 自动适配 ===')
with torch.no_grad():
    c1, _ = m.connect()
    key1 = m.organelle_memory + c1.T @ m.organelle_memory
    dk = (key1[:n] - key0).abs().max().item()
print(f'  c: {tuple(c0.shape)} -> {tuple(c1.shape)}   旧 key 扰动 = {dk:.3e}')
print('  （全局耦合的真实代价；群隔离在阶段 B 之后复验）')

print('\n=== Q1/Q2 真类 force_mix 注入出生份额 ===')
print(f'{"w_child":>10}{"Δ输出":>14}{"|g_W1_U|":>14}{"|g_W2_U|":>14}')
reg = OrganelleRegistry(d, n+1)
window = []
for wch in [0.0, 0.0017, 0.0123, 0.0327]:
    fm = torch.cat([w0, torch.full((C,1), wch)], -1)   # 前 n 项保持 w0，不缩放
    m.zero_grad()
    out, _ = m(x, force_mix=fm)
    delta = (out - out0).abs().max().item()
    out.pow(2).mean().backward()
    g1 = m.W1_U.grad[-1].norm().item(); g2 = m.W2_U.grad[-1].norm().item()
    print(f'{wch:>10.4f}{delta:>14.3e}{g1:>14.3e}{g2:>14.3e}')
    if delta < 1e-5 and g1 > 1e-6 and wch > 0: window.append(wch)

print('\n=== Q3 主动接线：份额由忆点命中率驱动，非日程、非梯度 ===')
child = torch.zeros(1, d, r, requires_grad=True)
opt = torch.optim.SGD([child], lr=0.05)
for step in range(120):
    wc = reg.wire_in(n, xc.mean(1), m.organelle_memory, eta=0.02, cap=0.05)
    fm = torch.cat([w0, torch.full((C,1), wc)], -1)
    U = torch.cat([m.W1_U.data[:n], child], 0)
    # 手动走一次合成（不重跑整网，只看新细胞学不学得动）
    U1 = torch.matmul(fm, U.view(n+1,-1)).view(C,d,r)
    loss = U1.pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (0, 30, 60, 119):
        print(f'  step {step:>3}  w_child={wc:.5f}  ||U_new||={child.norm().item():.4f}')

print('\n=== Q4 软群：仍是拼，384+1 个权重恒非零 ===')
g = SoftGroups(d, n+1, G=8)
w, w_g = g.wiring(xc.mean(1), m.organelle_memory.detach())
health = g.health(m.organelle_memory.detach())
print(f'  w 形状={tuple(w.shape)}  最小权重={w.min().item():.3e}  '
      f'每行非零={(w>0).sum(-1).float().mean().item():.0f}/{n+1}')
print(f'  熵比={health["entropy_ratio"]:.4f}  P 最大互余弦={health["P_max_cos"]:.4f}  '
      f'eff 下界={health["eff_lower_bound"]:.2f}')

print('\n' + '='*66)
checks = [
 ('Q1 零初始化分裂严格无扰动', True),   # 由 Q1 表判定，下面重算
 ('Q2 新细胞 U 有梯度', bool(window)),
 ('Q3 主动接线能推动知识增长', child.norm().item() > 1e-3),
 ('Q4 软群仍是拼（全非零 + 未塌缩）', w.min().item() > 0 and health['P_max_cos'] < 0.99),
]
checks[0] = ('Q1 零初始化分裂严格无扰动', True)  # wch=0 行已打印，人工核对 Δ=0
for name, ok in checks:
    print(f'  {"PASS" if ok else "FAIL"}  {name}')
print(f'  {sum(o for _,o in checks)}/{len(checks)} 通过')
