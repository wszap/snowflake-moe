# -*- coding: utf-8 -*-
"""任务1.5 诊断：打印每段 k / 奖励均值 / cv / 段平均loss，验证动态性。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from marvis_moe import Config, make_synthetic
from marvis_moe_v7 import train_v7

d, E, S, L, K = 16, 64, 2, 2, 8
cfg = Config(d=d, h=32, E=E, S=S, L=L, topk=6, out_dim=K)
X, y = make_synthetic(n=4096, d=d, k=K, seed=2026)
perm = torch.randperm(X.shape[0])
X, y = X[perm], y[perm]
Xtr, ytr, Xva, yva = X[:3072], y[:3072], X[3072:], y[3072:]

model, h, council = train_v7(cfg, Xtr, ytr, Xva, yva, council_mode='learned',
                             seed=2026, epochs=12, batch_size=256, lr=2e-3)

print("\n=== 每段治理事件（k 动态检查）===")
ks = []
for ev in h['events']:
    act = ev[2]
    ks.append(act['k'])
    print(f"step={ev[1]:4d} lbda={act['lbda']:.3f} cap={act['cap']:.3f} k={act['k']:3d} edu={act['edu']:.3f}")
print(f"k 序列: {ks}  唯一值={len(set(ks))}")

print("\n=== 奖励统计 ===")
rew = np.array(h['rewards'])
print(f"rewards n={len(rew)}  mean={rew.mean():.4f} std={rew.std():.4f} min={rew.min():.4f} max={rew.max():.4f}")

print("\n=== 训练曲线 ===")
print(f"val_acc 最终={h['val_acc'][-1]:.4f} 曲线={[round(v,4) for v in h['val_acc']]}")
print(f"cv 最终={h['final_cv']:.4f} 活跃={h['final_active']:.2%}")
print(f"loss 首={h['loss'][0]:.4f} 尾={h['loss'][-1]:.4f}")
