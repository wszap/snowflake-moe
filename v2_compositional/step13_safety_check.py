import torch, torch.nn.functional as F
from snowflake_B import SnowflakeB
from organelle_groups import SoftGroups

torch.manual_seed(0)
d, n, G = 256, 384, 8
m = SnowflakeB(d, n_organelles=n, h=128, rank=4, seq_len=128, chunk_size=1, temp=5.0, seed=0)
x = torch.randn(128, d); xsum = m._chunk_view(x).mean(1)
c, _ = m.connect()
key = m.organelle_memory + c.T @ m.organelle_memory
w0 = F.softmax(xsum @ key.T / d**0.5 * m.temp, dim=-1).detach()

def eff(w): return (1.0 / w.pow(2).sum(-1).mean()).item()
def ratio(w): return (w / w0).min().item(), (w / w0).max().item()

g = SoftGroups(d, n, G=G, tau=0.10)
g.init_from_memories(m.organelle_memory)

print(f'{"P 状态":<22}{"eff":>9}{"比值min":>11}{"比值max":>11}{"全非零":>9}')
print('-'*64)
cases = [('正常', lambda: None),
         ('P×1e6', lambda: g.P.data.mul_(1e6)),
         ('P×1e-6', lambda: g.P.data.mul_(1e-6)),
         ('P 全零', lambda: g.P.data.zero_()),
         ('P 全塌成一行', lambda: g.P.data.copy_(g.P.data[0].expand_as(g.P.data))),
         ('P 每行随机重排', lambda: g.P.data.copy_(torch.randn_like(g.P)))]
for name, fn in cases:
    fn()
    w, _ = g.compose_prior(w0, xsum, m.organelle_memory.detach(), beta=0.1)
    lo, hi = ratio(w)
    print(f'{name:<22}{eff(w):>9.2f}{lo:>11.4f}{hi:>11.4f}{str(bool((w>0).all())):>9}')

print(f'\n原 wiring eff = {eff(w0):.2f}')
print('理论夹子 beta=0.1 → 比值 ∈ (0.909, 1.100), eff ∈ [0.83,1.21]×eff_orig')
print('判据：六档全部落在夹子内、全非零 → P 任意失控都 fail-safe to 原拼装')

# ---- 群塌成「选」的方向：A.max 能到多少（没有 ent_floor 时）----
print('\n=== 无熔断时软归属能推到多尖 ===')
for tau in [0.10, 0.05, 0.02, 0.005]:
    gg = SoftGroups(d, n, G=G, tau=tau); gg.init_from_memories(m.organelle_memory)
    A = gg.membership(m.organelle_memory.detach())
    print(f'  tau={tau:<6} A.max={A.max(-1).values.mean().item():.4f}  '
          f'→ {"已接近选群 ⚠" if A.max(-1).values.mean().item()>0.9 else "仍是软群"}')
