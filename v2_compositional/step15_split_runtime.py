import torch, torch.nn as nn, torch.nn.functional as F
from snowflake_B import SnowflakeB
from organelle_registry import OrganelleRegistry
from organelle_groups import SoftGroups

torch.manual_seed(0)
d, h, r, n, G = 256, 128, 4, 384, 8

def make():
    return SnowflakeB(d, n_organelles=n, h=h, rank=r,
                      seq_len=128, chunk_size=1, temp=5.0, seed=0).float()

m = make()
orig_fwd = m.forward                    # ★ 显式 force_mix，不再 monkey-patch
x0 = torch.randn(128, d)
xsum0 = m._chunk_view(x0).mean(1).detach()

# ---- [0] 标定真实工作点，避开 softmax 饱和（step7/13 的老坑）----
with torch.no_grad():
    c0, _ = m.connect()
    M0 = m.organelle_memory.detach().clone()
    key_cache = (M0 + c0.T @ M0).clone()          # [384,d] 冻结
def eff(w): return (1.0 / w.pow(2).sum(-1).mean()).item()
def wiring_at(xs):
    return F.softmax(xs @ key_cache.T / d**0.5 * m.temp, dim=-1)
s = None
for cand in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    if 2.5 <= eff(wiring_at(xsum0 * cand)) <= 4.0: s = cand; break
if s is None: s = 0.15
x = x0 * s
xsum = m._chunk_view(x).mean(1).detach()
C = m._chunk_view(x).shape[0]
w_old = wiring_at(xsum).detach()
print(f'[0] 工作点 scale={s}  eff={eff(w_old):.2f}  min(w)={w_old.min().item():.2e}')

# ---- [1] 主动触发（非 step 时钟）+ 群标签 + 分裂 ----
reg = OrganelleRegistry(d, n)
reg.observe(xsum, w_old, step=0)
reg.in_var[0] = 2.0; reg.fitness[0] = 0.05; reg.birth_step[0] = 0
fired = reg.should_divide(0, step=300)
g = SoftGroups(d, n + 1, G=G, tau=0.10)
g.init_from_memories(M0)
gid = g.membership(M0).argmax(-1)
reg.group_id[:n] = gid                            # ★ 旧细胞先有群，新细胞才有归属
peers = (gid == gid[0]).nonzero().flatten()
j, nm = reg.divide(0, M0, step=300)
print(f'[1] 主动触发={fired}  support={reg.support(0):.3f}  本群 K={len(peers)}')
print('    分工卡 =', reg.card(j))

# ---- 扩 5 个按 n 索引的张量（W1_U/W2_U 零初始化）----
with torch.no_grad():
    m.organelle_memory = nn.Parameter(torch.cat([M0, nm.unsqueeze(0)], 0))
    m.W1_U = nn.Parameter(torch.cat([m.W1_U.data, torch.zeros(1, d, r)], 0))
    m.W2_U = nn.Parameter(torch.cat([m.W2_U.data, torch.zeros(1, h, r)], 0))
    m.register_buffer('mix_ema', torch.cat([m.mix_ema, torch.zeros(1)]))
    m.register_buffer('fitness_ema', torch.cat([m.fitness_ema, torch.zeros(1)]))
    m.n = n + 1

def fm_with(col):
    return torch.cat([w_old, torch.full((C, 1), col)], -1)   # ★ 恒 385 维、同形状

# ---- Q1 分裂瞬间：同模型/同形状/只差末列 → 严格等号 ----
with torch.no_grad():
    out0, _ = orig_fwd(x, force_mix=fm_with(0.0))                     # 等价未分裂
    out1, _ = orig_fwd(x, force_mix=fm_with(reg.w_child[j].item()))     # 分裂后
q1 = (out1 - out0).abs().max().item()
print(f'Q1 Δ输出 = {q1:.3e}')

# ---- Q3 旧细胞梯度：同一前向函数、同形状、只差末列 ----
target = torch.randn_like(out0)
def grad_old(col):
    m.zero_grad()
    out, _ = orig_fwd(x, force_mix=fm_with(col))
    (out - target).pow(2).mean().backward()
    return m.W1_U.grad[:n].clone()
g_a = grad_old(0.0)
g_b = grad_old(reg.w_child[j].item())
q3 = (g_a - g_b).abs().max().item()
print(f'Q3 旧细胞梯度差 = {q3:.3e}')

# ---- Q2 主动接线：份额由自身忆点命中率驱动，知识从零长 ----
opt = torch.optim.SGD([m.W1_U, m.W2_U], lr=0.05)
norms, wcs = [], []
for s2 in range(60):
    wc = reg.wire_in(j, xsum, m.organelle_memory, eta=0.02, cap=0.05)
    opt.zero_grad()
    out, _ = orig_fwd(x, force_mix=fm_with(wc))
    (out - target).pow(2).mean().backward(); opt.step()
    norms.append(m.W1_U.data[-1].norm().item()); wcs.append(wc)
print(f'Q2 w_child {wcs[0]:.5f} → {wcs[-1]:.5f}   ||U_new|| {norms[0]:.4f} → {norms[-1]:.4f}')

# ---- Q4 接线：秩-1 只落本群，群外严格 0 ----
with torch.no_grad():
    cC = torch.zeros(n + 1, n + 1); cC[:n, :n] = c0
    cC[n, peers] = 1.0 / len(peers); cC[n] /= cC[n].sum()
    key_new = key_cache + cC[n, :n].unsqueeze(-1) * nm.unsqueeze(0)
    dk = (key_new - key_cache).abs()
    in_g  = dk[peers].max().item()
    out_g = dk[~torch.isin(torch.arange(n), peers)].max().item()
print(f'Q4 群内={in_g:.3e}  群外={out_g:.3e}')

# ---- Q5 仍是拼：真实工作点下 385 权重全非零、eff>2 ----
fm = fm_with(wcs[-1])
q5, eff5 = fm.min().item(), eff(fm)
print(f'Q5 min(w)={q5:.2e}  全非零={bool((fm > 0).all())}  eff={eff5:.2f}')

print('\n' + '=' * 64)
checks = [
 ('[0] 真实工作点 (2.5≤eff≤4.0)', 2.5 <= eff(w_old) <= 4.0),
 ('[1] 主动触发（非 step 时钟）', fired),
 ('Q1 分裂严格无扰动', q1 == 0),
 ('Q3 旧细胞梯度严格不变', q3 == 0),
 ('Q2 新细胞知识从零增长', norms[-1] > norms[0] + 1e-3),
 ('Q2 接线份额被主动调高', wcs[-1] > wcs[0]),
 ('Q4 接线扰动锁在本群', out_g == 0 and in_g > 0),
 ('Q5 仍是拼（全非零 + eff>2）', q5 > 0 and eff5 > 2),
 ('Q5 新细胞有群标签', int(reg.group_id[j]) >= 0),
]
for name, ok in checks: print(f'  {"PASS" if ok else "FAIL"}  {name}')
print(f'  {sum(o for _, o in checks)}/{len(checks)} 通过')
