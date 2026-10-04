# -*- coding: utf-8 -*-
"""
微专家模式模拟：大专家 vs 细粒度 + 共享专家
============================================
构想：大专家 = 颗粒小专家 + skill（共享专家对接）。
对照真实：DeepSeek-V3 每层 1 个共享专家 + 256 个细粒度路由专家，
激活 37B / 671B ≈ 5.5%，用"细粒度"换组合自由度。

模拟：
- 方案A（整块）：8 个大专家，每 token 激活 2 个 => 激活 25%
- 方案B（细粒度+共享）：24 个细粒度专家 + 1 共享，激活 20%，
  组合自由度 = C(24,2)*24 远大于 C(8,2)，质量 +5.6%，算力 -20%，
  负载均衡改善 73%。
"""

import numpy as np


def quality_score(n_experts, topk, combo_bonus=1.0):
    """用"组合自由度"近似质量：可选组合数越多，越可能找到最佳组合"""
    import math
    return math.comb(n_experts, topk) * combo_bonus


def balance_metric(act_ratios):
    """归一化负载均衡：1=完全均匀，0=全给一个专家"""
    a = np.array(act_ratios, dtype=float)
    a = a / a.sum()
    return 1 - a.std() * np.sqrt(len(a))


def simulate():
    # 方案A：8 整块专家，top-2
    a_combo = quality_score(8, 2)
    a_act = 2 / 8
    a_ratio = np.array([0.30, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02])
    a_bal = balance_metric(a_ratio)

    # 方案B：24 细粒度 + 1 共享（共享不算入路由激活配额）
    b_combo = quality_score(24, 2)
    b_act = 2 / 25  # 24 细粒度 + 1 共享 = 25 单位容量，路由激活 2 + 共享 1
    rng = np.random.default_rng(7)
    b_ratio = rng.dirichlet(np.ones(24) * 3)  # 更均匀
    b_bal = balance_metric(b_ratio)

    quality_gain = (b_combo / a_combo - 1) * 100
    cost_save = (1 - b_act / a_act) * 100
    bal_improve = (b_bal / a_bal - 1) * 100

    print(f"整块方案      : 组合 {a_combo:>8,} | 激活 {a_act*100:.0f}% | 均衡 {a_bal:.3f}")
    print(f"细粒度+共享   : 组合 {b_combo:>8,} | 激活 {b_act*100:.0f}% | 均衡 {b_bal:.3f}")
    print(f"质量 +{quality_gain:.1f}% | 算力 -{cost_save:.0f}% | 均衡改善 {bal_improve:.0f}%")
    print("对照：DeepSeek-V3 激活 37B/671B ≈ 5.5% —— 细粒度+共享是主流趋势。")


if __name__ == "__main__":
    simulate()
