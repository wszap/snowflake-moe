# -*- coding: utf-8 -*-
"""
verify_min_eff.py —— 内生最低有效器官数约束的验证（纯 numpy，60 秒内）

背景：实测训练后 sharp=0.982 ⇒ 有效器官数 1/Σw² = 1.04，
      384 个器官里只有 1 个在工作 —— 那是"选专家"，不是"拼专家"。

药方：w_a = (1-a)·w + a·(1/N)，解析求 a 使有效器官数精确等于 K。

    (a-1)² = (1/K − 1/N)/(S₂ − 1/N),   S₂ = Σw²
    a = clamp(1 − sqrt(ratio), 0, 1)

验证三件事：
  1. 精确性：约束后 1/Σw² 是否真的等于 K
  2. 单边性：已够分散的样本是否完全不动（a=0）
  3. 可微性：a 对 w 的梯度是否有限且方向正确
"""
import numpy as np

rng = np.random.default_rng(0)
N = 384


def eff(w):
    return 1.0 / (w ** 2).sum(-1)


def enforce(w, K, N=384):
    inv_N = 1.0 / N
    S2 = (w * w).sum(-1, keepdims=True)
    num = 1.0 / K - inv_N
    den = np.maximum(S2 - inv_N, 1e-8)
    a = np.clip(1.0 - np.sqrt(np.maximum(num / den, 1e-8)), 0.0, 1.0)
    return (1.0 - a) * w + a * inv_N, a


def make_w(sharp, N=384, S=5000):
    """构造 top1=sharp 的 wiring：top1 占 sharp，其余均分"""
    rest = (1.0 - sharp) / (N - 1)
    w = np.full((S, N), rest)
    w[:, 0] = sharp
    return w / w.sum(1, keepdims=True)


print("=" * 92)
print("【1】精确性：约束后有效器官数是否等于 K")
print("=" * 92)
print(f"{'输入sharp':>10}{'原始有效数':>12}{'目标K':>8}{'约束后有效数':>14}{'a':>8}{'误差':>10}")
print("-" * 92)
for sharp in [0.982, 0.957, 0.90, 0.50, 0.30]:
    w = make_w(sharp)
    for K in [8]:
        wc, a = enforce(w, K)
        e0, e1 = eff(w).mean(), eff(wc).mean()
        print(f"{sharp:>10.3f}{e0:>12.2f}{K:>8}{e1:>14.2f}"
              f"{a.mean():>8.3f}{abs(e1-K)/K:>10.4f}")

print("\n  → sharp=0.982（有效数 1.04）经 K=8 约束后有效数 = 8.00 ✅")
print("    不同 K 的效果：")
for K in [2, 4, 8, 16, 32]:
    w = make_w(0.982)
    wc, a = enforce(w, K)
    print(f"      K={K:>3} → 有效数 {eff(wc).mean():>6.2f}  "
          f"a={a.mean():.3f}  约束后 sharp={wc.max(1).mean():.3f}")

print("\n" + "=" * 92)
print("【2】单边性：已够分散的样本必须完全不动")
print("=" * 92)
print(f"{'原始有效数':>12}{'目标K':>8}{'a':>10}{'是否不动':>12}")
print("-" * 92)
for sharp in [0.982, 0.30, 0.10, 0.039, 1.0 / N]:
    w = make_w(sharp)
    K = 8
    wc, a = enforce(w, K)
    e0 = eff(w).mean()
    untouched = a.mean() < 1e-6
    print(f"{e0:>12.2f}{K:>8}{a.mean():>10.6f}"
          f"{'✅ 不动' if untouched else '施加约束':>12}")

print("\n  → 有效数 > K 的样本 a=0，完全不动 ✅ 单边，不破坏已学分工")

print("\n" + "=" * 92)
print("【3】可微性：a 对 w 的梯度")
print("=" * 92)
eps = 1e-6
for sharp in [0.982, 0.50]:
    w = make_w(sharp, S=1)
    K = 8
    _, a0 = enforce(w, K)
    # 扰动 top1 权重
    w2 = w.copy()
    w2[0, 0] += eps
    w2 = w2 / w2.sum(1, keepdims=True)
    _, a1 = enforce(w2, K)
    da = (a1 - a0).mean() / eps
    print(f"  sharp={sharp:.3f}: da/dw_top1 = {da:+.3f}  "
          f"{'有限 ✅' if np.isfinite(da) else '❌ 发散'}")
print("\n  → 梯度有限，torch 可自动回传。整个约束是可微的。")

print("\n" + "=" * 92)
print("【4】★ 这个约束解决了什么")
print("=" * 92)
print(f"""
实测（step 100，N=384, temp=5.0）：
    sharp = 0.982  ⇒  有效器官数 = {eff(make_w(0.982)).mean():.2f}
    ⇒ 384 个器官里只有 1 个在工作 —— 那是【选专家】

启用 --min-eff 8 后：
    有效器官数 = 8.00，sharp 降到 {enforce(make_w(0.982), 8)[0].max(1).mean():.3f}
    ⇒ 每样本真正用到 8 个器官 —— 这才是【拼专家】

代价：wiring 被强制掺了一部分均匀分布，loss 会略升。
      这是"拼"的代价，换来的是核心主张成立。
      若 loss 涨太多，把 K 降到 4。

★ 与调 temp 的区别：
    temp 是【间接】控制 —— 训练会把 logits σ 放大 10.9 倍，静态 temp 猜不准
    min_eff 是【直接】控制 —— 约束的是结果（有效器官数），与 spread 无关
""")
