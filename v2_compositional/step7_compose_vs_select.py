import torch, torch.nn.functional as F
from snowflake_B import SnowflakeB

torch.manual_seed(0)
d, h, r, n = 256, 128, 4, 384          # 与 sh_6000 config 对齐
m = SnowflakeB(d, n_organelles=n, h=h, rank=r,
               seq_len=128, chunk_size=1, temp=5.0, seed=0)
x = torch.randn(128, d)
xc = m._chunk_view(x); C = xc.shape[0]
c, _ = m.connect()
key = m.organelle_memory + c.T @ m.organelle_memory
w = F.softmax(xc.mean(1) @ key.T / d**0.5 * m.temp, dim=-1)   # [C,384]

L = m.W1_U                                  # [384,d,r] 专家库
UA = torch.einsum('cn,ndr->cdr', w, L)     # A 拼：按输入合成新专家
UB = L.mean(0).expand(C, -1, -1)            # B 静态平均：无动态拼装
UC = L[w.argmax(-1)]                       # C 选：只取最强那一个

# 关键指标 1：合成专家是不是库里已有的？
FA = F.normalize(UA.reshape(C, -1), dim=-1)
FL = F.normalize(L.reshape(n, -1), dim=-1)
best = (FA @ FL.T).max(-1).values
print(f'Q1  合成专家 vs「最像的单个专家」余弦 = {best.mean():.4f} (max {best.max():.4f})')
print(f'    远 < 1 → 拼出来的是库里【不存在】的新专家，不是从库里挑')

# 关键指标 2：跨输入是否动态组合
def cross(U):
    f = F.normalize(U.reshape(U.shape[0], -1), dim=-1)
    M = f @ f.T
    off = ~torch.eye(U.shape[0], dtype=torch.bool)
    return M[off].mean().item()
print(f'Q2  跨 chunk 平均余弦： 拼 A={cross(UA):.4f}  静态 B={cross(UB):.4f}  选 C={cross(UC):.4f}')
print(f'    A 介于 B/C 之间且随输入变化 → 动态拼装确实在发生')

# 关键指标 3：选专家丢掉多少
print(f'Q3  ||UA-UC|| / ||UA|| = {((UA-UC).norm()/UA.norm()).item():.4f}')
print(f'    >0 → one-hot 丢失了其余 383 个专家的合成贡献')

# 关键指标 4：长尾质量
top4 = w.topk(4, -1).values
print(f'Q4  top1 份额={top4[:,0].mean():.4f}  top2~4={top4[:,1:].sum(-1).mean():.4f}  '
      f'其余380个={(1-top4.sum(-1)).mean():.4f}')
print(f'    长尾非零 → 拼装中每个专家都在贡献，砍掉即破坏拼')

print('\n' + '='*62)
print('  PASS 若 Q1<0.9 且 Q3>0：结构上「拼」不等于「选」')
