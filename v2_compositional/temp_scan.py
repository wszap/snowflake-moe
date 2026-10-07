# -*- coding: utf-8 -*-
"""
temp_scan.py —— 温度扫描（纯 numpy，沙箱可跑，60 秒内）

★ 核心发现：temp 必须随 N 缩放，规格的 temp=2.0 在 N=384 下会掉进
  【均匀平均陷阱】（病理1）—— 另一个极端。

为什么：
    logits ≈ N(0, σ²)，σ ∝ temp·|x|·|key|/√d
    sharp 由 N 个 logits 的 max 决定 ⇒ E[max] ≈ σ·√(2·ln N)
    N=8   → √(2·ln8)   = 2.04
    N=384 → √(2·ln384) = 3.45     ← 因子涨 1.69x

    ⇒ N 越大，同样的 temp 下 logits 越分散，但 N 也越大 ⇒ 均匀度竞争

实测（未训练初始状态）：
    N=8,   d=64,  temp=4.0 → sharp=0.505
    N=384, d=128, temp=2.0 → sharp=0.039   ← 过均匀，掉进陷阱
    N=384, d=128, temp=5.1 → sharp=0.30    ← 目标区

用法：
    python temp_scan.py
"""
import numpy as np


def sharp_of(N, d, temp, noise=0.5, seed=0, n_x=2048, band=0.5, width=0.1, td=4.0):
    r = np.random.default_rng(seed)
    # 红线6 规格初始化
    base = r.normal(size=d)
    base = base / np.linalg.norm(base) * 0.5 * np.sqrt(d)
    mem = base + r.normal(size=(N, d)) * noise

    mn = mem / np.linalg.norm(mem, axis=-1, keepdims=True)
    c = np.exp(-(((mn @ mn.T) - band) / width) ** 2)
    c = c / (c.sum(-1, keepdims=True) + td) * td
    key = mem + c.T @ mem

    x = r.normal(size=(n_x, d))
    lg = x @ key.T / np.sqrt(d) * temp
    w = np.exp(lg - lg.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    return dict(sharp=float(w.max(1).mean()),
                ent=float(-(w * np.log(w + 1e-9)).sum(1).mean()),
                max_ent=float(np.log(N)),
                std=float(lg.std()),
                top5=float(np.sort(w, 1)[:, -5:].sum(1).mean()))


def solve_temp(N, d, target=0.30, lo=0.05, hi=40.0, iters=40):
    for _ in range(iters):
        m = (lo + hi) / 2
        if sharp_of(N, d, m)["sharp"] > target:
            hi = m
        else:
            lo = m
    return (lo + hi) / 2


if __name__ == "__main__":
    print("=" * 92)
    print("【1】未训练初始 sharp vs temp —— 两个极端都要避开")
    print("=" * 92)
    print("""
    sharp → 1.0  ：赢者通吃（病理2），退化为离散选择
    sharp → 1/N  ：均匀平均（病理1），退化为单个平均专家
    目标区：0.2 ~ 0.4（连续组合，多个器官真正参与）
    """)
    print(f"{'N':>6}{'d':>6}{'temp':>7}{'sharp':>9}{'ent':>9}{'ent/lnN':>10}"
          f"{'top5质量':>10}{'判定':>16}")
    print("-" * 92)
    for N, d, t in [(8, 64, 4.0), (8, 64, 2.0), (8, 64, 1.8),
                    (384, 128, 2.0), (384, 128, 3.0), (384, 128, 5.0),
                    (384, 128, 8.0), (384, 256, 2.0), (384, 256, 5.0),
                    (512, 512, 2.0), (512, 512, 8.0)]:
        r = sharp_of(N, d, t)
        # 判据用 ent/lnN 与 top5，不用 sharp 绝对值：
        # sharp 的"均匀"基准 1/N 随 N 变，绝对值无法跨 N 比较。
        ent_ratio = r["ent"] / r["max_ent"]
        if r["sharp"] > 0.9:
            j = "赢者通吃 ⚠"
        elif ent_ratio > 0.85 or r["top5"] < 0.25:
            j = "过均匀（陷阱侧）⚠"
        elif 0.2 <= r["sharp"] <= 0.4 and r["top5"] > 0.45:
            j = "✅ 目标区"
        else:
            j = "可接受"
        print(f"{N:>6}{d:>6}{t:>7.2f}{r['sharp']:>9.4f}{r['ent']:>9.3f}"
              f"{r['ent']/r['max_ent']:>10.3f}{r['top5']:>10.3f}{j:>16}")

    print("\n" + "=" * 92)
    print("【2】各规模下让 sharp = 0.30 所需的 temp")
    print("=" * 92)
    print(f"{'N':>6}{'d':>6}{'temp*':>9}{'验证sharp':>12}{'规格temp=2.0下的sharp':>24}")
    print("-" * 92)
    for N, d in [(8, 64), (64, 128), (384, 128), (384, 256), (512, 512)]:
        t = solve_temp(N, d)
        s_star = sharp_of(N, d, t)["sharp"]
        s_2 = sharp_of(N, d, 2.0)["sharp"]
        print(f"{N:>6}{d:>6}{t:>9.3f}{s_star:>12.4f}{s_2:>24.4f}")

    print("""
★ 结论：规格的 temp=2.0 只在 N≈8 附近合适。
  N=384 时 temp=2.0 的初始 sharp=0.039 ≈ 1/N=0.0026 的 15 倍，
  虽然不算严格均匀，但处于【过均匀】侧，训练压力小、分工动力不足。

  建议正式训练扫 temp ∈ {2, 3, 5, 8}，以 wiring_variance 和 val_ppl 为判据。
""")

    print("=" * 92)
    print("【3】⚠ 一个必须澄清的观测假象")
    print("=" * 92)
    print("""
冒烟日志 step 0 就显示 sharp=1.000，容易被读成"初始化就 one-hot"。
但脚本的打印在【epoch 循环之后】，所以 step 0 的数值是
训练完 1 个 epoch（4096/256 = 16 个 batch）之后的状态，不是初始值。

实测初始 sharp（N=8, d=64, temp=4.0）= 0.505 —— 相当均匀。
    → 训练 1 个 epoch → 1.000
    → 之后 299 步维持 0.997

⇒ 用户说的"早熟硬化"方向是对的，机制是【任务诱导】：
  合成任务是 8 簇 / 每簇一个真专家 / sep=2.0，最优解【就是】one-hot。
  模型一眼看穿，1 个 epoch 就收敛到最优解。

⇒ 这不代表架构有问题。真实 LM 没有这么极端的簇结构。
   但必须在真实任务上观测 sharp，不能只看合成任务。
""")
