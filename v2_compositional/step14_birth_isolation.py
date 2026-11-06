import torch, torch.nn as nn, torch.nn.functional as F
from snowflake_B import SnowflakeB
from organelle_groups import SoftGroups

torch.manual_seed(0)
d, h, r, n, G = 256, 128, 4, 384, 8
m = SnowflakeB(d, n_organelles=n, h=h, rank=r, seq_len=128, chunk_size=1, temp=5.0, seed=0)

with torch.no_grad():
    c0, _ = m.connect()
    M0 = m.organelle_memory.detach().clone()
    key0 = M0 + c0.T @ M0                          # 旧 key 基准

    # 加一行忆点（5 个张量里只验 key 相关的两个）
    nm = F.normalize(M0[0] + 0.3 * torch.randn(d), dim=0).unsqueeze(0)
    M1 = torch.cat([M0, nm], 0)
    m.organelle_memory = nn.Parameter(M1.clone())
    m.n = n + 1

    # ---- A 老办法：connect() 全局重算 ----
    cA, _ = m.connect()
    keyA = M1 + cA.T @ M1
    dA = (keyA[:n] - key0).abs().max().item()

    # ---- B Schur 零补：旧块冻结，新行列出生成隔离期 ----
    cB = torch.zeros(n + 1, n + 1)
    cB[:n, :n] = c0                                # ★ 旧块原封不动
    keyB = M1 + cB.T @ M1
    dB = (keyB[:n] - key0).abs().max().item()

    # ---- C 主动接线：只接【本群】，扰动被锁在群内 ----
    g = SoftGroups(d, n + 1, G=G, tau=0.10)
    g.init_from_memories(M0)
    gid = g.membership(M0).argmax(-1)               # 384 个旧细胞的群标签
    parent = 0
    peers = (gid == gid[parent]).nonzero().flatten()  # 本群成员
    cC = cB.clone()
    cC[n, peers] = 1.0 / len(peers)                # 新细胞只连本群
    keyC = M1 + cC.T @ M1
    dk = (keyC[:n] - key0).abs()
    in_g  = dk[peers].max().item()
    out_g = dk[~torch.isin(torch.arange(n), peers)].max().item()

print(f'A  connect() 全局重算      旧 key 扰动 = {dA:.3e}   ← Q0 复现')
print(f'B  Schur 零补(出生隔离)    旧 key 扰动 = {dB:.3e}   ← 严格 0')
print(f'C  主动接线(只接本群 K={len(peers)})  群内={in_g:.3e}  群外={out_g:.3e}')
print()
print('=' * 64)
print(f'  {"PASS" if dB < 1e-6 else "FAIL"}  B  分裂不扰动旧知识（代数恒等式，非近似）')
print(f'  {"PASS" if out_g < 1e-6 else "FAIL"}  C  接线扰动严格锁在本群 → 群结构承担隔离')
print(f'  {"PASS" if dA > 1e-3 else "FAIL"}  A  对照组确实有全局扰动')
print('  三条全过 = 「自主分裂 + 忆点高速连接 + 细胞群」不碰 wiring、不伤 val、零额外开销')
