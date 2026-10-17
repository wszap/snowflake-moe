"""
挂谷翻转塞 · 换挂载位置（沿序列段）的数学自检
=============================================
纯 numpy，不需要 GPU，10 秒内跑完。

验证五条性质：
    ① 零初始化 ⇒ 严格恒等（加了绝不会变差）
    ② 因果性   ⇒ 位置 p 的输出不依赖 p 之后的任何输入（不泄漏未来）
    ③ 旋转保范 ⇒ carry 范数严格不变（两个支点稳住，永不放大缩小）
    ④ 长程有界 ⇒ 传很多段也不爆炸/消失
    ⑤ 非零 θ   ⇒ 确实改变方向（模块是活的，不是摆设）

② 是新挂载点【最关键】的一条：v6 挂在逐 token 上，结构上天然不泄漏；
   现在沿位置传递，必须重新证明它不偷看未来。

③ 是修 bug 后重测的：旧版用 Householder 插值 y = x − 2α(x·u)u，
   沿轴分量被乘 (1−2α)，中间角度会收缩（实测比值 0.87）。
   改用 Givens 旋转（绕轴转，刚体运动）后应严格保范。
"""

import numpy as np

rng = np.random.default_rng(0)


def givens_rotate(x, u, v, theta):
    """绕 (u,v) 平面旋转 theta —— 严格保范"""
    a = np.sum(x * u, axis=-1)
    b = np.sum(x * v, axis=-1)
    c, s = np.cos(theta), np.sin(theta)
    return (x
            + (a * (c - 1.0) - b * s)[..., None] * u
            + (a * s + b * (c - 1.0))[..., None] * v)


def segment_flipper(x, anchors, K=8, theta=None, inj=0.0, out=0.0,
                    keep=1.0, temp=1.0, n_seg=8):
    """沿序列段推进的翻转塞（numpy 复现 kakeya_seg.SegmentKakeyaFlipper）"""
    B, T, d = x.shape
    S = n_seg
    seg_len = T // S
    A_n = anchors / np.linalg.norm(anchors, axis=-1, keepdims=True)
    ths = np.zeros(K) if theta is None else np.asarray(theta)

    h = np.zeros((B, d))
    outs = []
    for s in range(S):
        seg = x[:, s * seg_len:(s + 1) * seg_len, :]
        outs.append(seg + out * h[:, None, :])     # 读：用上一段留下的 h
        g = seg.mean(axis=1)                       # 攒：本段摘要
        scores = g @ A_n.T
        idx = np.argsort(-scores / temp, axis=-1)[:, : 2 * K]
        P = A_n[idx]                               # (B, 2K, d)
        for k in range(K):
            u = P[:, 2 * k, :]
            v = P[:, 2 * k + 1, :]
            v = v - np.sum(v * u, axis=-1, keepdims=True) * u   # Gram-Schmidt
            nv = np.linalg.norm(v, axis=-1, keepdims=True)
            v = np.where(nv > 1e-12, v / np.maximum(nv, 1e-12), 0.0)
            h = givens_rotate(h, u, v, ths[k])
        h = h * keep
        h = h + inj * g
    return np.concatenate(outs, axis=1)


def main():
    B, T, d, n = 4, 128, 128, 384
    anchors = rng.normal(size=(n, d)) * 0.1
    x = rng.normal(size=(B, T, d))
    K = 8

    print("=" * 66)
    print("  挂谷翻转塞 · 沿序列段（Givens 版）—— 数学自检")
    print("=" * 66)
    results = []

    # ---- ① 零初始化 ⇒ 严格恒等 ----
    y = segment_flipper(x, anchors, K=K, theta=np.zeros(K),
                        inj=0.0, out=0.0, keep=1.0)
    err = np.abs(y - x).max()
    ok = err == 0.0
    results.append(ok)
    print(f"\n① 零初始化严格恒等")
    print(f"   max|y - x| = {err:.3e}   "
          f"{'✅ 精确为 0' if ok else '❌ 有偏差'}")

    # ---- ② 因果性（新挂载点最关键）----
    t = 64
    x2 = x.copy()
    x2[:, t:, :] += 1.0
    kw = dict(theta=np.full(K, 0.7), inj=0.5, out=0.5, keep=0.99)
    y1 = segment_flipper(x, anchors, K=K, **kw)
    y2 = segment_flipper(x2, anchors, K=K, **kw)
    diff_before = np.abs(y1[:, :t, :] - y2[:, :t, :]).max()
    diff_after = np.abs(y1[:, t:, :] - y2[:, t:, :]).max()
    ok = diff_before == 0.0 and diff_after > 0
    results.append(ok)
    print(f"\n② 因果性（不偷看未来）")
    print(f"   位置 < {t} 的差异 = {diff_before:.3e}  "
          f"{'✅ 完全不受影响' if diff_before == 0 else '❌ 泄漏了未来'}")
    print(f"   位置 >= {t} 的差异 = {diff_after:.3e}  "
          f"{'✅ 正常响应' if diff_after > 0 else '❌ 模块没起作用'}")

    # ---- ③ 旋转保范（修 bug 后重测）----
    h = rng.normal(size=(B, d))
    n0 = np.linalg.norm(h, axis=-1)
    A_n = anchors / np.linalg.norm(anchors, axis=-1, keepdims=True)
    for trial, ang in enumerate([0.0, 0.3, 0.785, 1.5, 3.14159]):
        hh = h.copy()
        scores = hh @ A_n.T
        idx = np.argsort(-scores, axis=-1)[:, : 2 * K]
        P = A_n[idx]
        for k in range(K):
            u = P[:, 2 * k, :]
            v = P[:, 2 * k + 1, :]
            v = v - np.sum(v * u, axis=-1, keepdims=True) * u
            nv = np.linalg.norm(v, axis=-1)
            degen = nv < 1e-6
            v = np.where(nv[..., None] > 1e-12,
                         v / np.maximum(nv[..., None], 1e-12), 0.0)
            hh = givens_rotate(hh, u, v, np.where(degen, 0.0, ang))
        r = np.linalg.norm(hh, axis=-1) / n0
        print(f"   θ={ang:>6.3f}  |h| 比值 = {r.mean():.10f}")
    # 正式判定用一个中间角度
    hh = h.copy()
    for k in range(K):
        u = A_n[rng.integers(0, n, size=B)]
        v = A_n[rng.integers(0, n, size=B)]
        v = v - np.sum(v * u, axis=-1, keepdims=True) * u
        nv = np.linalg.norm(v, axis=-1)
        degen = nv < 1e-6
        v = np.where(nv[..., None] > 1e-12,
                     v / np.maximum(nv[..., None], 1e-12), 0.0)
        hh = givens_rotate(hh, u, v, np.where(degen, 0.0, 0.7))
    ratio = np.linalg.norm(hh, axis=-1) / n0
    ok = np.abs(ratio - 1.0).max() < 1e-10
    results.append(ok)
    print(f"\n③ Givens 旋转保范（两个支点稳住）")
    print(f"   θ=0.700  |h| 比值 = {ratio.mean():.10f}  "
          f"{'✅ 严格守恒' if ok else '❌ 范数变了'}")

    # ---- ④ 长程有界 ----
    S_long, inj, keep = 200, 0.1, 0.99
    hs = np.zeros((B, d))
    norms = []
    for s in range(S_long):
        g = rng.normal(size=(B, d)) * 0.1
        for k in range(K):
            u = A_n[rng.integers(0, n, size=B)]
            v = A_n[rng.integers(0, n, size=B)]
            v = v - np.sum(v * u, axis=-1, keepdims=True) * u
            nv = np.linalg.norm(v, axis=-1)
            degen = nv < 1e-6
            v = np.where(nv[..., None] > 1e-12,
                         v / np.maximum(nv[..., None], 1e-12), 0.0)
            hs = givens_rotate(hs, u, v, np.where(degen, 0.0, 0.7))
        hs = hs * keep + inj * g
        norms.append(np.linalg.norm(hs, axis=-1).max())
    norms = np.array(norms)
    ok = np.isfinite(norms).all() and norms.max() < 50
    results.append(ok)
    print(f"\n④ 长程传递有界（{S_long} 段）")
    print(f"   carry 范数：起 {norms[0]:.4f}  末 {norms[-1]:.4f}  "
          f"最大 {norms.max():.4f}  "
          f"{'✅ 不爆炸' if ok else '❌ 发散'}")

    # ---- ⑤ 非零 θ ⇒ 模块是活的 ----
    y0 = segment_flipper(x, anchors, K=K, theta=np.zeros(K),
                         inj=0.0, out=0.0)
    yA = segment_flipper(x, anchors, K=K,
                         theta=np.full(K, 0.7), inj=0.3, out=0.3)
    delta = np.abs(yA - y0).max()
    ok = delta > 1e-3
    results.append(ok)
    print(f"\n⑤ 模块是活的（θ/门非零时确有作用）")
    print(f"   max|y(θ=0.7) − y(θ=0)| = {delta:.3e}  "
          f"{'✅ 确实在翻转' if ok else '❌ 没反应'}")

    print("\n" + "=" * 66)
    print(f"  汇总：{sum(results)}/{len(results)} 通过")
    print("=" * 66)
    return all(results)


if __name__ == "__main__":
    ok = main()
    print("\n结论：" + ("数学全部成立，可以上训练" if ok
                       else "有条目未通过 —— 先别跑训练"))
