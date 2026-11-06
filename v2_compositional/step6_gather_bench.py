import torch, torch.nn.functional as F, time
from snowflake_B import SnowflakeB

torch.manual_seed(0)
d, h, r = 256, 128, 4
m = SnowflakeB(d, n_organelles=384, h=h, rank=r,
               seq_len=128, chunk_size=32, seed=0)
x = torch.randn(128, d)
xc = m._chunk_view(x)
C, P, _ = xc.shape
c, _ = m.connect()
key = m.organelle_memory + c.T @ m.organelle_memory
w = F.softmax(xc.mean(1) @ key.T / d**0.5 * m.temp, dim=-1)   # [C,384]

flat = m.W1_U.view(384, -1)
U_dense = torch.matmul(w, flat).view(C, d, r)

topv, topi = torch.topk(w, 4, dim=-1)
topv = topv / topv.sum(-1, keepdim=True)
U_gath = torch.einsum('ck,ckdr->cdr', topv, m.W1_U[topi])

print(f'Q1  稠密合成 vs topk(4)+gather 最大误差 = {(U_dense-U_gath).abs().max().item():.3e}')
print(f'    (wiring 尖锐时应当很小；这决定近似是否无损)')

for name, fn in [('dense', lambda: torch.matmul(w, flat)),
                 ('gather', lambda: torch.einsum('ck,ckdr->cdr', topv, m.W1_U[topi]))]:
    t = time.time()
    for _ in range(200): fn()
    print(f'Q2  {name:<7} 200 次耗时 = {time.time()-t:.4f}s')

spars = (w > 1e-3).sum(-1).float().mean().item()
print(f'Q3  wiring 每行 >1e-3 的个数 = {spars:.1f} / 384  → 实际稀疏度')
