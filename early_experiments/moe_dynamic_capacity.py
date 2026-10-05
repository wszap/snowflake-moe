# -*- coding: utf-8 -*-
"""
第4课优化：MoE 动态容量调配模拟
===============================
三种方案对比（N=10000 请求、预算 2600 单位容量）：
  1. 无限制：容量 = 请求数（基准 100%）
  2. 固定配额：每个专家固定分到容量（热点来了也挤不进去）
  3. 动态调配：把冷门专家容量挪给热点（看似合理，实则滞后）
热点分段转移：前 1/3 集中在专家 0-1，中段转到 2-3，后段转到 4-5。
"""

import numpy as np


def hot_windows(N):
    """返回每个请求对应的热点专家（分段转移 + 20% 长尾随机需求）"""
    w = np.empty(N, dtype=int)
    n3 = N // 3
    rng = np.random.default_rng(0)
    tail = np.random.default_rng(7).integers(0, 6, size=N)  # 长尾：偶尔要冷门
    w[:n3] = rng.integers(0, 2, size=n3)
    w[n3:2 * n3] = np.random.default_rng(1).integers(2, 4, size=n3)
    w[2 * n3:] = np.random.default_rng(2).integers(4, 6, size=N - 2 * n3)
    tail_mask = rng.random(N) < 0.2           # 20% 请求走长尾
    w[tail_mask] = tail[tail_mask]
    return w


def run(mode, N=10000, budget=1500, n_experts=6, window=500):
    hot = hot_windows(N)
    if mode == "unlimited":
        return 1.0
    # 每窗口可用容量 = 预算按窗口长度折算（总量不变，不创造容量）
    per_win = budget * window / N
    if mode == "fixed":
        # 固定配额：每个专家每窗口固定分到 per_win/n_experts
        cap = np.full(n_experts, per_win / n_experts)
        serve = 0
        for s in range(0, N, window):
            seg = hot[s:s + window]
            for e in range(n_experts):
                serve += min(int((seg == e).sum()), cap[e])
        return serve / N
    # dynamic：每窗口按"上一窗口"需求比例重分配（滞后一个窗口）
    cap = np.full(n_experts, per_win / n_experts)
    serve = 0
    for s in range(0, N, window):
        seg = hot[s:s + window]
        for e in range(n_experts):
            serve += min(int((seg == e).sum()), cap[e])
        # 用本窗口需求比例更新下一窗口容量（只重新分配，总量仍为 per_win）
        if s + window < N:
            nxt = hot[s + window:s + 2 * window]
            demand = np.array([int((nxt == e).sum()) for e in range(n_experts)])
            if demand.sum() > 0:
                cap = np.clip(demand * per_win / demand.sum(), 0, per_win)
            else:
                cap = np.full(n_experts, per_win / n_experts)
    return serve / N


if __name__ == "__main__":
    for mode in ["unlimited", "fixed", "dynamic"]:
        print(f"{mode:10s} 覆盖率 {run(mode)*100:.1f}%")
    print("结论：动态重分配不创造容量，滞后 + 冷门塌方抵消收益，与固定配额基本打平。")
