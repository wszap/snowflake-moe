import torch, torch.nn.functional as F
from snowflake_B import SnowflakeB

torch.manual_seed(0)
d, h, r, n, temp = 256, 128, 4, 384, 5.0        # 与 sh_6000 config 对齐
m = SnowflakeB(d, n_organelles=n, h=h, rank=r,
               seq_len=128, chunk_size=1, temp=temp, seed=0)
c, _ = m.connect()
key = m.organelle_memory + c.T @ m.organelle_memory
L = m.W1_U
FL = F.normalize(L.reshape(n, -1), dim=-1)
x0 = torch.randn(128, d)

print(f'{"scale":>8}{"eff":>9}{"top1":>9}{"top4":>9}{"长尾":>9}{"Q1cos":>10}{"Q3rel":>10}')
print('-'*66)
best = None
for s in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
    x = x0 * s
    xc = m._chunk_view(x); C = xc.shape[0]
    w = F.softmax(xc.mean(1) @ key.T / d**0.5 * temp, dim=-1)
    eff  = (1.0 / (w*w).sum(-1).mean()).item()
    t4   = w.topk(4, -1).values
    UA   = torch.einsum('cn,ndr->cdr', w, L)
    FA   = F.normalize(UA.reshape(C, -1), dim=-1)
    q1   = (FA @ FL.T).max(-1).values.mean().item()
    UC   = L[w.argmax(-1)]
    q3   = ((UA - UC).norm() / UA.norm()).item()
    print(f'{s:>8.2f}{eff:>9.2f}{t4[:,0].mean():>9.4f}{t4.sum(-1).mean():>9.4f}'
          f'{(1-t4.sum(-1)).mean():>9.4f}{q1:>10.4f}{q3:>10.4f}')
    if 2.5 <= eff <= 4.0 and best is None:
        best = (s, eff, q1, q3)

print('\n' + '='*66)
if best:
    s, eff, q1, q3 = best
    print(f'  真实工作点: scale={s}  eff={eff:.2f}  (训练末步 3.253)')
    print(f'  {"PASS" if q1 < 0.95 else "FAIL"}  Q1 合成专家不是库里任一个 (cos={q1:.4f})')
    print(f'  {"PASS" if q3 > 0.05 else "FAIL"}  Q3 选专家会丢失长尾贡献 (rel={q3:.4f})')
    print('  → 两项同时 PASS：结构上「拼」≠「选」，且长尾不可砍')
else:
    print('  未扫到 eff 2.5~4.0，把 scale 范围往小再扩')
