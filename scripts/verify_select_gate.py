"""验证核架选择门的因果性 —— numpy 复现 torch 同一套逻辑（含 clamp 修复）"""
import numpy as np
T, d, k = 128, 16, 32
rng = np.random.default_rng(0)
h2 = rng.normal(size=(T, d))
Wq = rng.normal(size=(d, d)) / np.sqrt(d)
Wk = rng.normal(size=(d, d)) / np.sqrt(d)
causal = np.triu(np.ones((T, T), bool), 1)

def softmax_rows(v):
    v = v - v.max(-1, keepdims=True); e = np.exp(v); return e / e.sum(-1, keepdims=True)

def select(h, clamp=True):
    s = (h @ Wq) @ (h @ Wk).T / np.sqrt(d)
    s = np.where(causal, -np.inf, s)
    ti = np.argsort(-s, axis=-1)[:, :k]
    tv = np.take_along_axis(s, ti, axis=-1)
    if clamp:                                   # ★ 修复
        ti = np.minimum(ti, np.arange(T)[:, None])
    w = softmax_rows(tv)
    return ti, w, (w[..., None] * h[ti]).sum(1)

print("=== 修复前 ===")
ti0, _, _ = select(h2, clamp=False)
print(f"  索引指向未来的条数 = {(ti0 > np.arange(T)[:,None]).sum()}  ❌")

print("\n=== 修复后 ===")
ti1, w1, o1 = select(h2, clamp=True)
print(f"  索引指向未来的条数 = {(ti1 > np.arange(T)[:,None]).sum()}  ✅")
print(f"  含 NaN = {np.isnan(w1).any()}  {'❌' if np.isnan(w1).any() else '✅'}")
print(f"  各行权重和全为1 = {np.allclose(w1.sum(-1),1)}")
print(f"  t=0 第一条权重 = {w1[0,0]:.6f} (应为1.0)")

h2b = h2.copy(); h2b[16:32] = rng.normal(size=(16,d))
_, _, o2 = select(h2b, clamp=True)
diff = np.abs(o1[:16]-o2[:16]).max()
print(f"\n  改未来16..31 → 过去0..15差异 = {diff:.2e}  {'✅ 无泄漏' if diff==0 else '❌'}")

_, _, o1n = select(h2, clamp=False)
_, _, o2n = select(h2b, clamp=False)
print(f"  修复前后输出是否等价 = {np.allclose(o1,o1n)} (clamp 只断图，不改数值)")
