import math, torch, torch.nn.functional as F
from snowflake_B import SnowflakeB
from organelle_groups import SoftGroups

torch.manual_seed(0)
d, n, G = 256, 384, 8
m = SnowflakeB(d, n_organelles=n, h=128, rank=4, seq_len=128, chunk_size=1, temp=5.0, seed=0)
M = F.normalize(m.organelle_memory.detach(), dim=-1)
c, _ = m.connect()

# ---- [1a] kNN 余弦图（topk=4 对应 target_partners=4）+ 对称化 ----
cos = M @ M.T
_, idx = torch.topk(cos, 5, dim=-1)                 # 含自己
A = torch.zeros_like(cos)
for i in range(n): A[i, idx[i]] = 1.0
A = ((A + A.T) > 0).float(); A.fill_diagonal_(0)   # ★ 对称化，负特征值消失
L = torch.diag(A.sum(-1)) - A
ev = torch.linalg.eigvalsh(L)
print('[1a] kNN 拉普拉斯前 12 =', [round(v,3) for v in ev[:12].tolist()])
print('     <1.0 的个数(自然社区数) =', int((ev < 1.0).sum()))

# ---- [1b] 旧二值化 c 作对照（说明为什么上轮得出"无群"）----
A01 = (c.abs() > c.abs().mean()).float()
A01 = (A01 + A01.T) / 2
L0 = torch.diag(A01.sum(-1)) - A01
ev0 = torch.linalg.eigvalsh(L0)
print(f'[1b] 二值化 c: 前3={[round(v,2) for v in ev0[:3].tolist()]}  '
      f'每行度≈{A01.sum(-1).mean():.0f}/{n}  → 稠密随机图，无社区')

# ---- [1c] 忆点初始化偏置 ----
off = cos[~torch.eye(n, dtype=torch.bool)]
print(f'[1c] 忆点余弦: mean={off.mean():.4f} std={off.std():.4f} '
      f'min={off.min():.4f} max={off.max():.4f}  (全正 → 共享 base 主簇)')

# ---- [2] 余弦 k-means ----
def kmeans_cos(M, G, iters=30):
    C = M[torch.randperm(M.shape[0])[:G]].clone()
    for _ in range(iters):
        a = (M @ C.T).argmax(-1)
        for g in range(G):
            if (a == g).any(): C[g] = F.normalize(M[a == g].mean(0), dim=0)
    return C, a
P, assign = kmeans_cos(M, G)
print(f'[2] 群大小 = {torch.bincount(assign, minlength=G).tolist()}')

# ---- [3] 软归属：随机 P vs 结构化 P ----
g = SoftGroups(d, n, G=G, tau=0.5, temp_g=2.0)
with torch.no_grad():
    Ag_rand = g.membership(M).max(-1).values.mean().item()
    g.P.copy_(P)
    Ag_struct = g.membership(M).max(-1).values.mean().item()
print(f'[3] A.max 均值: 随机P={Ag_rand:.4f}  k-means P={Ag_struct:.4f}  '
      f'(1/G={1/G:.4f})')
print('    判据: struct 明显高于 rand → 群质心承载了结构')

# ---- [4] 替换 vs 乘性先验 ----
x = torch.randn(128, d); xsum = m._chunk_view(x).mean(1)
key = m.organelle_memory + c.T @ m.organelle_memory
w_orig = F.softmax(xsum @ key.T / math.sqrt(d) * m.temp, dim=-1)
w_struct, w_g = g.wiring(xsum, m.organelle_memory.detach())
def eff(w): return (1.0 / w.pow(2).sum(-1).mean()).item()
w_mix = w_orig * (1.0 + 0.1 * w_struct); w_mix /= w_mix.sum(-1, keepdim=True)
print(f'[4] eff: w_orig={eff(w_orig):.2f}  w_struct(替换)={eff(w_struct):.2f}  '
      f'w_mix(乘性β=0.1)={eff(w_mix):.2f}')
print(f'    min(w_mix)={w_mix.min().item():.2e}  全非零={(w_mix>0).all().item()}')
print('    判据: eff(w_struct) >> eff(w_mix) ≈ eff(w_orig)  → 必须乘性，不能替换')
