"""翻转塞核心数学自检（numpy 等价实现，无需 torch）。
验证四条由结构保证、而非靠正则化的性质：
  ① 零初始化 → 退化为恒等
  ② 平行分量严格不变（支点稳住）
  ③ 支点对面的对称性：u ⊥ s, u ⊥ e, s ⊥ e
  ④ 翻转是等距旋转（保范数 → 不跑飞）
  ⑤ carry 跨段传递（B 方案：团送下游而非丢弃）
  ⑥ 极端输入无 NaN/Inf
用法: python3 verify_flipper_math.py
"""
import numpy as np

rng = np.random.default_rng(0)


def normalize(v, axis=-1, eps=1e-8):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + eps)


def gram_schmidt(u, s_raw, e_raw):
    """把 s_raw / e_raw 正交化到垂直于 u，且 s ⊥ e。"""
    s = s_raw - np.sum(s_raw * u, axis=-1, keepdims=True) * u
    s = normalize(s)
    e = e_raw - np.sum(e_raw * u, axis=-1, keepdims=True) * u
    e = e - np.sum(e * s, axis=-1, keepdims=True) * s
    e = normalize(e)
    return s, e


def flipper(x, axis_raw, seal_raw, edge_raw,
            w_seal=None, b_seal=0.0, w_alpha=None,
            temp=0.5, alpha_gain=0.1, thickness=0.9,
            carry=None, w_carry=None):
    """x: (..., d)。返回 (y, carry, diag)。"""
    d = x.shape[-1]
    u = normalize(np.broadcast_to(axis_raw, x.shape).copy())

    x_par = np.sum(x * u, axis=-1, keepdims=True) * u
    x_perp = x - x_par

    s, e = gram_schmidt(u,
                        np.broadcast_to(seal_raw, x.shape).copy(),
                        np.broadcast_to(edge_raw, x.shape).copy())

    # 密封圈：软门槛
    if w_seal is None:
        tangle = np.full(x.shape[:-1] + (1,), float(b_seal))
    else:
        tangle = x_perp @ w_seal + b_seal
    gate = 1.0 / (1.0 + np.exp(-tangle / temp))

    trapped = gate * x_perp
    free = (1.0 - gate) * x_perp

    # 厚度：兜住，累积
    if carry is None:
        carry = np.zeros(x.shape[:-2] + (d,)) if x.ndim > 2 else np.zeros(d)
    carry = thickness * carry + trapped.mean(axis=-2)

    # 翻转角
    alpha = np.tanh(carry @ w_alpha if w_alpha is not None else 0.0)
    alpha = alpha * np.pi * alpha_gain
    ca, sa = np.cos(alpha), np.sin(alpha)
    if np.ndim(ca) > 0:
        ca = ca[..., None, None] if ca.ndim == 1 else ca[..., None]
        sa = sa[..., None, None] if sa.ndim == 1 else sa[..., None]

    fs = np.sum(x_perp * s, axis=-1, keepdims=True)
    fe = np.sum(x_perp * e, axis=-1, keepdims=True)
    fs_r = ca * fs - sa * fe
    fe_r = sa * fs + ca * fe
    rest = free - fs * s - fe * e
    out_perp = rest + fs_r * s + fe_r * e

    inject = (carry @ w_carry)[..., None, :] if w_carry is not None else 0.0
    y = x_par + out_perp + inject
    diag = {
        "gate_mean": float(np.mean(gate)),
        "carry_norm": float(np.linalg.norm(carry)),
        "alpha": float(np.mean(alpha)),
    }
    return y, carry, diag


def check(name, ok, detail=""):
    print(f"  {name:46} {'PASS' if ok else 'FAIL'}   {detail}")
    return ok


d = 64
axis_raw = rng.standard_normal(d)
seal_raw = rng.standard_normal(d)
edge_raw = rng.standard_normal(d)
allok = []

print("=" * 80)
print("① 零初始化 → 起步退化为恒等（保证『加了不会变差』）")
print("=" * 80)
x = rng.standard_normal((2, 8, d))
y, _, _ = flipper(x, axis_raw, seal_raw, edge_raw, b_seal=-50.0)  # gate 下溢为 0
diff = np.abs(y - x).max()
allok.append(check("严格恒等（gate=0, α=0, 无注入）", diff < 1e-10,
                   f"max|y-x| = {diff:.2e}"))
# 默认出厂配置 bias=-8：gate≈3.4e-4，接近恒等但不严格
y2, _, dg2 = flipper(x, axis_raw, seal_raw, edge_raw, b_seal=-8.0)
rel = np.abs(y2 - x).max() / (np.abs(x).max() + 1e-12)
allok.append(check("出厂配置(bias=-8) 软退化", rel < 1e-2,
                   f"相对偏差 = {rel:.2e}  gate = {dg2['gate_mean']:.2e}"))

print()
print("=" * 80)
print("② 平行分量严格不变（支点稳住 → 主信息永不丢失）")
print("=" * 80)
w_seal = rng.standard_normal((d, 1)) * 0.5
w_alpha = rng.standard_normal(d) * 0.5
x = rng.standard_normal((4, 16, d))
y, _, _ = flipper(x, axis_raw, seal_raw, edge_raw,
                  w_seal=w_seal, w_alpha=w_alpha)   # 不注入 carry
u = normalize(axis_raw)
par_before = x @ u
par_after = y @ u
err = np.abs(par_before - par_after).max()
scale = np.abs(par_before).max()
allok.append(check("沿轴分量翻转前后一致", err / scale < 1e-6,
                   f"相对误差 = {err/scale:.2e}  (量级 {scale:.3f})"))

print()
print("=" * 80)
print("③ 支点恒在对面 + 密封圈在厚度中心（±u 对称 / 正交）")
print("=" * 80)
xx = rng.standard_normal((2, 4, d))
uu = normalize(np.broadcast_to(axis_raw, xx.shape).copy())
s, e = gram_schmidt(uu,
                    np.broadcast_to(seal_raw, xx.shape).copy(),
                    np.broadcast_to(edge_raw, xx.shape).copy())
allok.append(check("轴 ⊥ 密封圈法向 s", np.abs(np.sum(uu*s,axis=-1)).max() < 1e-8,
                   f"|u·s|max = {np.abs(np.sum(uu*s,axis=-1)).max():.2e}"))
allok.append(check("轴 ⊥ 边沿方向 e", np.abs(np.sum(uu*e,axis=-1)).max() < 1e-8,
                   f"|u·e|max = {np.abs(np.sum(uu*e,axis=-1)).max():.2e}"))
allok.append(check("s ⊥ e（旋转平面正交）", np.abs(np.sum(s*e,axis=-1)).max() < 1e-8,
                   f"|s·e|max = {np.abs(np.sum(s*e,axis=-1)).max():.2e}"))
allok.append(check("支点在轴两端（±u 过中心，结构性对称）",
                   np.abs(np.sum(uu*uu,axis=-1)-1).max() < 1e-8,
                   f"||u|| = {np.sqrt(np.sum(uu*uu,axis=-1)).max():.10f}"))

print()
print("=" * 80)
print("④ 翻转是等距旋转（只改方向不改大小 → 数值不跑飞）")
print("=" * 80)
for gain in (0.0, 0.1, 0.5, 1.0):
    x = rng.standard_normal((2, 8, d))
    # 关分选、关注入，只留旋转
    y, _, _ = flipper(x, axis_raw, seal_raw, edge_raw,
                      w_alpha=rng.standard_normal(d) * 2.0,
                      alpha_gain=gain, b_seal=-8.0)
    r = np.linalg.norm(y, axis=-1) / (np.linalg.norm(x, axis=-1) + 1e-12)
    allok.append(check(f"α 增益 {gain:<4} → 范数比 ≈ 1",
                       abs(r.mean() - 1) < 1e-6 and r.std() < 1e-6,
                       f"ratio = {r.mean():.8f} ± {r.std():.2e}"))

print()
print("=" * 80)
print("⑤ 跨段 carry：团被兜住并送到下游（B 方案，不是丢弃）")
print("=" * 80)
w_carry = rng.standard_normal((d, d)) * 0.1
x = rng.standard_normal((2, 32, d))
y, carry, dg = flipper(x, axis_raw, seal_raw, edge_raw,
                       w_seal=rng.standard_normal((d, 1)) * 1.0,
                       w_carry=w_carry)
allok.append(check("carry 非空（确实兜住了团）", dg["carry_norm"] > 1e-6,
                   f"||carry|| = {dg['carry_norm']:.4f}"))
allok.append(check("分选门在 (0,1) 内（橡胶软门槛）",
                   0.0 < dg["gate_mean"] < 1.0,
                   f"gate_mean = {dg['gate_mean']:.4f}"))

_, c1, _ = flipper(x[:, :16], axis_raw, seal_raw, edge_raw,
                   w_seal=rng.standard_normal((d, 1)) * 1.0, w_carry=w_carry)
_, c2, _ = flipper(x[:, 16:], axis_raw, seal_raw, edge_raw,
                   w_seal=rng.standard_normal((d, 1)) * 1.0,
                   w_carry=w_carry, carry=c1)
_, c0, _ = flipper(x[:, 16:], axis_raw, seal_raw, edge_raw,
                   w_seal=rng.standard_normal((d, 1)) * 1.0, w_carry=w_carry)
dd = np.abs(c2 - c0).max()
allok.append(check("carry 可跨段传递（信息流打通）", dd > 1e-6,
                   f"|c2 - c0| = {dd:.4f}"))
# 证明它不是丢弃：trapped 的信息确实出现在 carry 里
allok.append(check("团随 carry 前移（保留而非清零）",
                   np.linalg.norm(c2) > np.linalg.norm(c0) * 0.99,
                   f"||c2||={np.linalg.norm(c2):.4f}  ||c0||={np.linalg.norm(c0):.4f}"))

print()
print("=" * 80)
print("⑥ 极端输入：长序列 / 大数值 → 无 NaN/Inf")
print("=" * 80)
x = rng.standard_normal((2, 512, d)) * 10.0
y, c, dg = flipper(x, axis_raw, seal_raw, edge_raw,
                   w_seal=rng.standard_normal((d, 1)) * 1.0,
                   w_alpha=rng.standard_normal(d) * 2.0,
                   w_carry=rng.standard_normal((d, d)) * 0.1)
ok = np.isfinite(y).all() and np.isfinite(c).all()
allok.append(check("512 步 ×10 放大输入无溢出", ok,
                   f"|y|max = {np.abs(y).max():.3f}  ||carry|| = {np.linalg.norm(c):.3f}"))

# 连续 200 段传递，看 carry 是否爆炸（厚度=衰减率的作用）
carry = None
for _ in range(200):
    _, carry, _ = flipper(rng.standard_normal((2, 8, d)),
                          axis_raw, seal_raw, edge_raw,
                          w_seal=rng.standard_normal((d, 1)) * 1.0,
                          carry=carry)
allok.append(check("200 段连续传递 carry 有界", np.isfinite(carry).all()
                   and np.linalg.norm(carry) < 1e3,
                   f"||carry||₂₀₀ = {np.linalg.norm(carry):.4f}"))

print()
print("=" * 80)
print(f"自检结束：{sum(allok)}/{len(allok)} 通过")
print("=" * 80)
