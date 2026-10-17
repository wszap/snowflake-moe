"""
挂谷翻转塞 —— 数学自检（纯 numpy，无需 torch/GPU，约 5 秒）
==============================================================

验证 6 件事，第【4】条是决定性实验：
    ① 零初始化 → 严格恒等（加了不会变差）
    ② α=1 → 严格正交（等距：只改方向，不改大小）
    ③ Cartan–Dieudonné：≤ d 个反射可表示任意正交变换（构造性验证）
    ④ ★ 固定轴【不提秩】；检索轴【才提秩】（秩提升的真正来源）
    ⑤ 梯度能流动到 α
    ⑥ 参数 / FLOPs 对比 dense FFN
"""

import numpy as np

np.random.seed(0)
OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    flag = "✅" if cond else "❌"
    print(f"  [{flag}] {name}  {detail}")


def num_rank(M, tol=1e-5):
    """数值秩：奇异值 > tol * σ_max 的个数"""
    s = np.linalg.svd(np.asarray(M, dtype=np.float64), compute_uv=False)
    if len(s) == 0 or s[0] <= 0:
        return 0
    return int((s > tol * s[0]).sum())


def reflect(X, U, alpha):
    """一次翻转：y = x - 2α (x·u) u
    X : (N, d)    U : (N, d) 已归一化    alpha : 标量"""
    proj = np.sum(X * U, axis=1, keepdims=True) * U
    return X - 2.0 * alpha * proj


def topk_idx(scores, K):
    """每行取 top-K 的下标"""
    if K >= scores.shape[1]:
        return np.argsort(-scores, axis=1)
    part = np.argpartition(-scores, K - 1, axis=1)[:, :K]
    return part


def kakeya_numpy(X, anchors, K, alpha, diag_gate=None):
    """完整前向：检索 K 个轴 → 依次翻转 → 对角门
    X : (N, d)   anchors : (n, d)"""
    A = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-12)
    scores = X @ A.T                       # (N, n)
    idx = topk_idx(scores, K)              # (N, K)
    y = X.copy()
    for k in range(K):
        U = A[idx[:, k]]                   # (N, d) 每个 token 自己的轴
        y = reflect(y, U, alpha[k])
    if diag_gate is not None:
        y = y * (1.0 + np.tanh(diag_gate))
    return y


print("=" * 74)
print("  挂谷翻转塞 · 数学自检（numpy）")
print("=" * 74)

d, K, N_ANCH = 128, 8, 384

# =====================================================================
print("\n【1】零初始化 → 严格恒等（起步时不能改变任何东西）")
# =====================================================================
X = np.random.randn(64, d)
anchors = np.random.randn(N_ANCH, d) * 0.1
y0 = kakeya_numpy(X, anchors, K, np.zeros(K), np.zeros(d))
err = np.abs(y0 - X).max()
check("α=0 且门=0 时输出 == 输入", err < 1e-12, f"max|y-x| = {err:.2e}")

# =====================================================================
print("\n【2】α=1 → 严格正交（等距：只改方向，不改大小）")
# =====================================================================
y1 = kakeya_numpy(X, anchors, K, np.ones(K), np.zeros(d))
r_in = np.linalg.norm(X, axis=1)
r_out = np.linalg.norm(y1, axis=1)
ratio = float(np.mean(r_out / (r_in + 1e-12)))
check("范数保持 |y|/|x| ≈ 1", abs(ratio - 1.0) < 1e-8, f"ratio = {ratio:.12f}")

u = np.random.randn(d)
u = u / np.linalg.norm(u)
H = np.eye(d) - 2.0 * np.outer(u, u)
check("Householder 正交 HᵀH = I",
      np.allclose(H.T @ H, np.eye(d), atol=1e-10),
      f"误差 {np.abs(H.T @ H - np.eye(d)).max():.2e}")
check("Householder 自逆 H² = I",
      np.allclose(H @ H, np.eye(d), atol=1e-10))

# =====================================================================
print("\n【3】Cartan–Dieudonné：≤ d 个反射可表示任意正交变换")
# =====================================================================
def householder_decompose(Q):
    dd = Q.shape[0]
    A = Q.astype(np.float64).copy()
    parts = []
    for k in range(dd - 1):
        a = A[k:, k]
        nrm = np.linalg.norm(a)
        e1 = np.zeros_like(a)
        e1[0] = 1.0
        v = a - nrm * e1
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v = v / nv
        H = np.eye(len(a)) - 2.0 * np.outer(v, v)
        A[k:, k:] = H @ A[k:, k:]
        parts.append((k, v))
    return parts, A


def embed(parts, dd):
    M = np.eye(dd)
    for k, v in parts:
        H = np.eye(dd)
        H[k:, k:] -= 2.0 * np.outer(v, v)
        M = H @ M
    return M


for dd in [8, 32, 128]:
    Rm = np.random.randn(dd, dd)
    Qm, _ = np.linalg.qr(Rm)
    parts, Ares = householder_decompose(Qm)
    M = embed(parts, dd)
    resid = np.abs(M @ Qm - np.eye(dd)).max()
    n_extra = int((np.diag(Ares) < 0).sum())
    total = len(parts) + n_extra
    check(f"d={dd}: 反射数 ≤ d", total <= dd,
          f"{len(parts)}+{n_extra}={total} 个，残差 {resid:.2e}")
    check(f"d={dd}: 复合精确还原 Q", resid < 1e-8, f"残差 {resid:.2e}")

# =====================================================================
print("\n【4】★ 决定性实验：固定轴【不提秩】/ 检索轴【才提秩】")
print("      （正交变换保秩；秩的提升只能来自「轴随输入变化」）")
# =====================================================================
N_TOK, R_LOW = 512, 4
Z = np.random.randn(N_TOK, R_LOW)
W = np.random.randn(R_LOW, d)
Xlow = Z @ W                               # 低秩 delta（模拟 rank=4 的细胞器输出）
r0 = num_rank(Xlow)
print(f"      输入 delta 的数值秩 = {r0}   （被 rank=4 锁死）")

# ---- A) 固定轴：所有 token 共用同一组轴 ----
axes = np.random.randn(K, d)
axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
y_fixed = Xlow.copy()
for k in range(K):
    U = np.tile(axes[k], (N_TOK, 1))       # 每个 token 同一个轴
    y_fixed = reflect(y_fixed, U, 1.0)
r_fixed = num_rank(y_fixed)
check("固定轴：秩【几乎不变】", r_fixed <= r0 + 1,
      f"rank {r0} → {r_fixed}")

# ---- B) 检索轴：每个 token 自己挑轴 ----
y_ret = kakeya_numpy(Xlow, anchors, K, np.ones(K), np.zeros(d))
r_ret = num_rank(y_ret)
check("检索轴：秩【大幅提升】", r_ret > r0 * 3,
      f"rank {r0} → {r_ret}")

print("\n      K 扫描（检索轴模式的数值秩，输入秩=4，d=128）：")
rows = []
for kk in [4, 8, 16, 32, 64, 128]:
    yk = kakeya_numpy(Xlow, anchors, kk, np.ones(kk), np.zeros(d))
    rk = num_rank(yk)
    rows.append((kk, rk))
    bar = "█" * max(1, rk * 40 // d)
    print(f"        K={kk:>3}   秩 = {rk:>3} / {d}   {bar}")
best = max(r for _, r in rows)
print(f"      ⇒ 最大可达秩 = {best} / {d}"
      f"（{'满秩' if best >= d * 0.95 else '未达满秩'}）")

# =====================================================================
print("\n【5】梯度能流动到 α")
# =====================================================================
u5 = np.random.randn(d)
u5 /= np.linalg.norm(u5)
x5 = np.random.randn(d)
g_up = np.random.randn(d)                  # 上游梯度
# y = x - 2α(x·u)u  ⇒  ∂L/∂α = ⟨g_up, -2(x·u)u⟩
dalpha = float(np.dot(g_up, -2.0 * np.dot(x5, u5) * u5))
check("∂L/∂α 非零（能学）", abs(dalpha) > 1e-9, f"∂L/∂α = {dalpha:.4e}")

# =====================================================================
print("\n【6】参数 / FLOPs 对比 dense FFN（每 token）")
# =====================================================================
p_kak = K + d
p_ffn = d * 4 * d + 4 * d * d
f_retr = N_ANCH * d
f_refl = K * 3 * d
f_kak = f_retr + f_refl
f_ffn = d * 4 * d + 4 * d * d
print(f"      挂谷翻转塞  K={K}")
print(f"        参数   {p_kak:>10,}     （α {K} + 对角门 {d}，轴复用忆点）")
print(f"        FLOPs  {f_kak*2:>10,}     （检索 {f_retr*2:,} + 反射 {f_refl*2:,}）")
print(f"      dense FFN   d={d}, 4d={4*d}")
print(f"        参数   {p_ffn:>10,}")
print(f"        FLOPs  {f_ffn*2:>10,}")
print(f"      ⇒ 参数省 {p_ffn/max(p_kak,1):.0f}x ，FLOPs 省 {f_ffn/max(f_kak,1):.1f}x")
check("参数显著少于 dense FFN", p_kak < p_ffn, f"{p_kak} vs {p_ffn}")
check("FLOPs 显著少于 dense FFN", f_kak < f_ffn, f"{f_kak} vs {f_ffn}")

print("\n      K 对 FLOPs 的影响（d=128, 检索 384 轴）：")
for kk in [8, 16, 32, 64, 128]:
    fk = f_retr + kk * 3 * d
    print(f"        K={kk:>3}  FLOPs {fk*2:>8,}   （dense 的 {fk/f_ffn*100:>5.1f}%）")

# =====================================================================
print("\n" + "=" * 74)
n_ok = sum(OK)
print(f"  自检结果：{n_ok}/{len(OK)} 通过")
print("=" * 74)
if n_ok < len(OK):
    print("  ❌ 有未通过项，别急着跑训练")
else:
    print("  ✅ 全部通过，数学性质成立")
    print()
    print("  ★ 记住第【4】条的结论（这条最关键）：")
    print("     秩的提升【不是】反射带来的 —— 反射是正交变换，保秩。")
    print("     秩的提升来自【每个 token 检索到不同的轴】，")
    print("     于是低秩的 delta 被送到彼此不同的方向，整体覆盖维度扩大。")
    print("     这就是挂谷：参数极少（体积→0），却能覆盖所有方向（维数=n）。")
