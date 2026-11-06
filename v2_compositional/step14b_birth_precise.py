import torch, torch.nn as nn, torch.nn.functional as F
from snowflake_B import SnowflakeB
from organelle_groups import SoftGroups

torch.manual_seed(0)
d, h, r, n, G = 256, 128, 4, 384, 8

def run(dtype):
    m = SnowflakeB(d, n_organelles=n, h=h, rank=r,
                   seq_len=128, chunk_size=1, temp=5.0, seed=0).to(dtype=dtype)
    with torch.no_grad():
        c0, _ = m.connect()
        M0 = m.organelle_memory.detach().clone()
        key0 = M0 + c0.T @ M0                              # 旧 key 基准

        nm = F.normalize(M0[0] + 0.3*torch.randn(d, dtype=dtype), dim=0).unsqueeze(0)
        M1 = torch.cat([M0, nm], 0)
        m.organelle_memory = nn.Parameter(M1.clone()); m.n = n + 1

        # ---- A 对照：connect() 全局重算 ----
        cA, _ = m.connect()
        dA = ((M1 + cA.T @ M1)[:n] - key0).abs().max().item()

        # ---- B 隔离期：秩-1 增量式，c[n,:]=0 → 零次运算 ----
        cB = torch.zeros(n+1, n+1, dtype=dtype); cB[:n,:n] = c0
        assert cB[n].abs().max().item() == 0 and cB[:,n].abs().max().item() == 0
        keyB_old = key0 + cB[n,:n].unsqueeze(-1) * M1[n]     # ★ 秩-1
        keyB_new = M1[n] + cB[:,n] @ M1
        dB = (keyB_old - key0).abs().max().item()

        # ---- C 主动接线：只接本群，秩-1 只落 peers ----
        g = SoftGroups(d, n+1, G=G, tau=0.10).to(dtype=dtype)
        g.init_from_memories(M0)
        gid = g.membership(M0).argmax(-1)
        peers = (gid == gid[0]).nonzero().flatten()
        cC = cB.clone(); cC[n, peers] = 1.0/len(peers)
        cC[n] /= cC[n].sum()                                # 行归一化，对齐 connect 语义
        keyC_old = key0 + cC[n,:n].unsqueeze(-1) * M1[n]     # ★ 秩-1
        dk = (keyC_old - key0).abs()
        in_g  = dk[peers].max().item()
        out_g = dk[~torch.isin(torch.arange(n), peers)].max().item()

    print(f'[{str(dtype):<14}] A全局={dA:.3e}  B隔离={dB:.3e}  '
          f'C群内={in_g:.3e}  群外={out_g:.3e}  K={len(peers)}')
    print(f'    {"PASS" if dB == 0 else "FAIL"}  B  分裂严格不扰动旧 key')
    print(f'    {"PASS" if out_g == 0 else "FAIL"}  C  接线扰动严格锁在本群')
    print(f'    {"PASS" if dA > 1e-3 else "FAIL"}  A  对照组确有全局扰动')
    print(f'    {"PASS" if in_g > 0 else "FAIL"}  C  本群内确实建立了新连接（有意结果）')
    print(f'    新细胞 key_new[n] = M_new[n] ? {(keyB_new - M1[n]).abs().max().item()==0}')

run(torch.float32)
run(torch.float64)
