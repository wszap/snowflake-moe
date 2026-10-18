# -*- coding: utf-8 -*-
"""验证 v8 显存安全阀：一次性 gather 与 逐 slot 两条路径必须数值等价。

  背景：旧版 snowflake_v4.py:216 用 chunk_size=1（每 token 独立自选专家），
        而 v8 默认 chunk=32（32 个 token 共用一组）——对比不公平。
        为了让 chunk=1 能跑（否则 E1[topi] 约 537MB 会爆显存），
        composable_expert.py 加了自动安全阀：超预算时改逐 slot 计算。
        本脚本证明这个切换【不改变结果】。

  用法： python3 verify_v8_lowmem.py
"""
import torch
from composable_expert import ComposableExpert

torch.manual_seed(0)
d, n, k, h_cell, h = 64, 128, 8, 16, 32

m = ComposableExpert(d=d, n_organelles=n, h=h, h_cell=h_cell, k=k,
                     seq_len=32, chunk_size=1, seed=0)
x = torch.randn(4 * 32, d)          # N=128, seq_len=32, chunk=1 -> C=128

m.mem_budget_mb = 1e9               # 路径A：预算无限 -> 一次性 gather
oA, iA = m(x)
m.mem_budget_mb = 0.0               # 路径B：预算为 0 -> 逐 slot
oB, iB = m(x)

print("=" * 66)
print("  v8 显存安全阀等价性验证")
print("=" * 66)
print("  chunk=1 时 chunk 数 C =", iA["wiring"].shape[0])
print("  拼装张量占用 need     = %.1f MB" % iA["mem_need_mb"])
print()
print("  路径A（一次性 gather） lowmem_used =", iA["lowmem_used"])
print("  路径B（逐 slot）      lowmem_used =", iB["lowmem_used"])
print()
diff = (oA - oB).abs().max().item()
print("  max|路径A - 路径B| = %.3e" % diff)
print()
ok = diff < 1e-5
print("  %s  两条路径数值等价，安全阀不改变结果" % ("✅" if ok else "❌"))
print("=" * 66)

# ---- 另验：拼 k 个的输出确实随 k 变化（证明拼装真的参与计算）----
print()
print("  附加：拼装是否真的参与前向（gate=1 下改 k 应改变输出）")
outs = {}
for kk in (1, 2, 4, 8):
    mm = ComposableExpert(d=d, n_organelles=n, h=h, h_cell=h_cell, k=kk,
                          seq_len=32, chunk_size=1, seed=0)
    with torch.no_grad():
        mm.gate.fill_(1.0)
    o, _ = mm(x)
    outs[kk] = o
base = outs[1]
for kk in (2, 4, 8):
    dd = (outs[kk] - base).abs().max().item()
    print("    k=%d vs k=1: max差 = %.4f  %s"
          % (kk, dd, "（参与计算）" if dd > 1e-6 else "（没参与！可疑）"))
