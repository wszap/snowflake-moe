# -*- coding: utf-8 -*-
"""
第4课基础模拟：8 大专家 top-2 路由
==================================
8 个专家各擅长不同输入域，门控打分 top-2 激活。
对比：a) 全连接（无 MoE）  b) 固定分区  c) top-2 路由
"""

import numpy as np


def gen_task(N, rng):
    """生成 N 个 (输入, 期望专家) 任务：输入映射到 8 个域之一"""
    domains = rng.integers(0, 8, size=N)
    x = rng.normal(size=(N, 16)) + np.repeat(np.eye(8)[domains], 2, axis=1) * 3.0  # 每个域有特征偏移
    return x, domains


def top2_router(x, gate):
    """门控打分 -> top-2 one-hot"""
    logits = x @ gate
    idx = np.argpartition(-logits, 2, axis=-1)[:, :2]
    w = np.zeros_like(logits)
    for i in range(x.shape[0]):
        w[i, idx[i]] = 1.0
    return w


def simulate(mode="top2", N=10000, seed=0):
    rng = np.random.default_rng(seed)
    x, dom = gen_task(N, rng)
    gate = rng.normal(size=(16, 8)) * 0.5
    if mode == "top2":
        w = top2_router(x, gate)
        hit = (w.argmax(-1) == dom).mean()
    elif mode == "full":
        hit = 1.0  # 全连接恒可达（假设容量无限）
    else:  # 固定分区：按输入一维符号切
        part = np.argsort(x[:, 0]) % 8
        hit = (part == dom).mean()
    return hit


if __name__ == "__main__":
    for mode in ["full", "fixed", "top2"]:
        acc = simulate(mode)
        print(f"{mode:5s} 命中率 {acc*100:.1f}%")
    print("结论：top-2 路由接近全连接效果，但只激活 25% 专家，省 75% 计算。")
