# -*- coding: utf-8 -*-
"""
verify_clone.py —— numpy 验证「复制-联结」的有效秩

★ 上一版 bug：加了 W1_base（全秩随机矩阵），把差异全掩盖了。
  本版只测【拼专家贡献项】Σ_i w_i (W1_U[i] @ W1_V)，不含 base。

★ 关键结构
    x → xA = x @ U_team      [P, r]
    x → hh = xA @ W1_V       [P, h]
    复合变换 = U_team @ W1_V，形状 [d, h]
    秩上限 = min(eff_total × rank, h, d)
"""
import numpy as np

def eff_rank(M, tol=1e-2):
    s = np.linalg.svd(M, compute_uv=False)
    if s[0] <= 0:
        return 0, s
    return int((s > s[0] * tol).sum()), s

def build(n, d, h, r, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, d, r)) / np.sqrt(d),
            rng.standard_normal((r, h)) / np.sqrt(r))

def sparse_w(n, k, rng, pool=None):
    pool = np.arange(n) if pool is None else pool
    idx = rng.choice(pool, size=min(k, len(pool)), replace=False)
    w = np.zeros(n); w[idx] = 1.0 / len(idx)
    return w

def team_mat(W1_U, W1_V, w, spec_w, link=1.0):
    U_self = np.einsum('n,ndr->dr', w, W1_U)
    U_clone = np.einsum('n,ndr->dr', spec_w, W1_U)
    return (U_self + link * U_clone) @ W1_V

def main():
    n, d, r = 384, 128, 4
    rng = np.random.default_rng(1)
    k_w = 3

    print("=" * 70)
    print("  ① 有效秩：clone 能否提升【单次改造能力】")
    print("=" * 70)
    for h in (32, 64, 128):
        W1_U, W1_V = build(n, d, h, r, seed=0)
        print(f"\n  ── h={h}（复合矩阵 [d,h]=[{d},{h}]，秩上限 min(eff·r, h)）──")
        print(f"  {'配置':<30} {'有效秩':>8} {'上限':>6}")
        print("  " + "-" * 48)
        w = sparse_w(n, k_w, rng)
        M0 = team_mat(W1_U, W1_V, w, np.zeros(n), 0.0)
        rk0, _ = eff_rank(M0)
        print(f"  {'v3 基线（wiring 激活 3）':<26} {rk0:>8} {min(k_w*r, h):>6}")
        w_idx = set(np.where(w > 0)[0])
        pool = np.array([i for i in range(n) if i not in w_idx])
        for k_s in (3, 8, 16, 32):
            spec = sparse_w(n, k_s, rng, pool)
            Mt = team_mat(W1_U, W1_V, w, spec, 1.0)
            rk, _ = eff_rank(Mt)
            cap = min((k_w + k_s) * r, h)
            print(f"  {'+clone 复制 %2d 个不同标本'%k_s:<26} {rk:>8} {cap:>6}")
        spec_same = w.copy()
        rk_same, _ = eff_rank(team_mat(W1_U, W1_V, w, spec_same, 1.0))
        print(f"  {'对照 复制相同标本':<26} {rk_same:>8} {min(k_w*2*r, h):>6}")

    print()
    print("=" * 70)
    print("  ② 真正的天花板：h（隐层宽度）")
    print("=" * 70)
    print(f"""
  复合变换 U_team @ W1_V 形状 [d,h]，秩【硬上限 = h】。
    当前 h=32  ⇒ 无论复制多少标本，秩最多 32
    h=64      ⇒ 最多 64
    h=128     ⇒ 最多 128（= d，与 dense 同量级）

  ⇒ clone 能把秩从 12 提到 32（满），但【卡在 h=32】。
    想再往上，必须同时把 h 加大。
    这是 clone 与 h 的【联动关系】，不是 clone 单独能解决的。

  dense 对照：d=288，FFN 中间层 4×288=1152，秩上限 288。
""")

    print("=" * 70)
    print("  ③ 组合多样性 / 无泄漏")
    print("=" * 70)
    print(f"""
  离散配对数：{n} × {n} = {n*n:,}（v3 只有 {n}）
  无泄漏：spec_w 来自当前 token 的 x，chunk_size=1 ⇒ 结构上不可能看未来
          仍需在 A10 实测 leak_check=0

  ★ 诚实的边界
    组合空间【维数】不增加（U_self/U_clone 同在 span(W1_U)，
    缩放不改变 span，上限仍是 min(n-1, d·r)=383，B 型本就满维）。
    增加的是【有效秩】和【离散配置多样性】。
""")

if __name__ == "__main__":
    main()
