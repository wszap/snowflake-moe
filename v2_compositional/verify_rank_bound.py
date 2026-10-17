# -*- coding: utf-8 -*-
"""
确认真正的秩瓶颈：是 rank（低秩因子）还是 eff（激活几个器官）？

结构：x → xA = x@U_team [P,r] → hh = xA@W1_V [P,h]
     复合变换 = U_team @ W1_V，形状 [d,h]
     秩 ≤ min(rank(U_team), rank(W1_V)) ≤ min(r, r) = r
"""
import numpy as np

def rank_of(M, tol=1e-3):
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > s[0]*tol).sum()) if s[0] > 0 else 0

def test(d=128, h=32, n=384, k_w=3, seed=0):
    print("="*66)
    print("  复合变换 U_team@W1_V 的实际秩")
    print("  (有效秩判据：奇异值 > 最大值的 0.1%)")
    print("="*66)
    rng = np.random.default_rng(seed)
    print(f"\n  d={d} h={h} n={n} wiring激活k={k_w}")
    print(f"  {'rank':>6} {'无clone':>8} {'+clone(16)':>10} {'秩上限':>8}")
    print("  " + "-"*40)
    for r in (4, 8, 16, 32, 64):
        W1_U = rng.standard_normal((n,d,r))/np.sqrt(d)
        W1_V = rng.standard_normal((r,h))/np.sqrt(r)
        idx = rng.choice(n, k_w, replace=False)
        w = np.zeros(n); w[idx] = 1/k_w
        U_self = np.einsum('n,ndr->dr', w, W1_U)
        M0 = U_self @ W1_V

        pool = np.array([i for i in range(n) if i not in set(idx)])
        sid = rng.choice(pool, 16, replace=False)
        s = np.zeros(n); s[sid] = 1/16
        U_clone = np.einsum('n,ndr->dr', s, W1_U)
        M1 = (U_self + 1.0*U_clone) @ W1_V

        cap = min(r, h, d)
        print(f"  {r:>6} {rank_of(M0):>8} {rank_of(M1):>10} {cap:>8}")
    print(f"""
  ⇒ 秩 = rank（低秩因子宽度），与 eff、与 clone 都无关。
    clone 不改变秩上限。

  为什么？U_team 是 [d, r] 矩阵，秩天然 ≤ r。
  n_cells 个 cell 并联求和，秩最多 n_cells × r。
""")

def test_cells(d=128, h=32, n=384, r=4, k_w=3, seed=0):
    print("="*66)
    print("  n_cells 并联对秩的贡献")
    print("="*66)
    rng = np.random.default_rng(seed)
    print(f"\n  rank={r}  {'n_cells':>8} {'复合秩':>8} {'上限':>8}")
    print("  " + "-"*32)
    for nc in (1,2,4,8):
        Ms = []
        for c in range(nc):
            W1_U = rng.standard_normal((n,d,r))/np.sqrt(d)
            W1_V = rng.standard_normal((r,h))/np.sqrt(r)
            idx = rng.choice(n, k_w, replace=False)
            w = np.zeros(n); w[idx]=1/k_w
            Ms.append(np.einsum('n,ndr->dr', w, W1_U) @ W1_V)
        Msum = sum(Ms)/len(Ms)
        print(f"  {r:>6} {nc:>8} {rank_of(Msum):>8} {min(nc*r, h, d):>8}")
    print(f"""
  ⇒ 并联 n_cells 个不同参数的 cell，秩可叠加到 n_cells×r
    但上限仍卡在 h（当前 h=32）
    n_cells=4, r=4 → 秩 16，卡在 h=32 之下
""")

def flops_note():
    print("="*66)
    print("  B 型 FLOPs 与 rank 的关系（文档给定）")
    print("="*66)
    print("""
    r=4  P=32 → 0.664x  ✅
    r=8  P=32 → 1.078x  ⚠
    r=16 P=32 → 1.906x  ❌

  ⇒ rank 加倍，合成项 n·d·r/P 线性涨，FLOPs 迅速超过 dense。
    这就是核心矛盾：
      要秩 → 加 rank → FLOPs 涨 → 失去效率优势
""")

if __name__ == "__main__":
    test()
    print()
    test_cells()
    print()
    flops_note()
