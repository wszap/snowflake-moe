import torch, torch.nn.functional as F
from snowflake_B import SnowflakeB
from organelle_groups import SoftGroups

torch.manual_seed(0)
d, n, G = 256, 384, 8
m = SnowflakeB(d, n_organelles=n, h=128, rank=4,
               seq_len=128, chunk_size=1, temp=5.0, seed=0)
M = F.normalize(m.organelle_memory.detach(), dim=-1)
Mc = F.normalize(M - M.mean(0, keepdim=True), dim=-1)   # ★ 去共享 base

def kmeans_cos(X, G, iters=30):
    C = X[torch.randperm(X.shape[0])[:G]].clone()
    for _ in range(iters):
        a = (X @ C.T).argmax(-1)
        for g in range(G):
            if (a == g).any(): C[g] = F.normalize(X[a == g].mean(0), dim=0)
    return C

print(f'{"center":>7}{"tau":>7}{"A.max":>10}{"最大群占比":>12}{"空群数":>8}')
print('-' * 48)
for center in [False, True]:
    X = Mc if center else M
    P = kmeans_cos(X, G)
    for tau in [0.5, 0.2, 0.1, 0.05]:
        g = SoftGroups(d, n, G=G, tau=tau)
        with torch.no_grad(): g.P.copy_(P)
        A = g.membership(M, center=center)
        sizes = torch.bincount(A.argmax(-1), minlength=G).float()
        print(f'{str(center):>7}{tau:>7.2f}{A.max(-1).values.mean().item():>10.4f}'
              f'{sizes.max().item()/n:>12.3f}{(sizes==0).sum().item():>8}')
print('\n判据: 找到一组 A.max>0.4 且 最大群占比<0.5 且 无空群 → 群质心承载了结构')
