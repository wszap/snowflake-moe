# -*- coding: utf-8 -*-
"""阶段 A/B/C 新模式快速冒烟：1 epoch 合成 + 1 epoch MNIST 前向，验证不崩且机制生效。"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from marvis_moe import Config, make_synthetic
from marvis_moe_v7 import train_v7, LearnedCouncil
from marvis_moe import MoELayer

ok = 0

# ---- 合成数据 1 epoch 冒烟：全部新模式 ----
cfg = Config(d=16, h=32, E=64, S=2, L=2, topk=6, out_dim=8)
X, y = make_synthetic(n=4096, d=16, k=8, seed=2026)
perm = np.random.RandomState(2026).permutation(X.shape[0])
X, y = X[perm], y[perm]
Xtr, ytr, Xva, yva = X[:3072], y[:3072], X[3072:], y[3072:]

modes = ['learned_ablate_loss', 'learned_ablate_cv', 'learned_ablate_k',
         'learned_capdrop', 'learned_tokendrop',
         'learned_no_edu', 'learned_fixed_edu', 'learned_adaptive_edu']
for m in modes:
    model, h, _ = train_v7(cfg, Xtr, ytr, Xva, yva, council_mode=m, seed=2026,
                           epochs=1, batch_size=512, lr=2e-3,
                           fixed_cap=1.2, use_token_dropping=(m == 'learned_tokendrop'))
    va = h['val_acc'][-1]
    assert va > 0.0, f"{m} val_acc 异常"
    ok += 1
    print(f"  smoke OK {m}: val_acc={va:.4f} cv={h['final_cv']:.4f}")

# ---- Token Dropping 机制验证：drop_ratio 随 cap 单调（cap 大则丢弃少）----
layer = MoELayer(cfg)
x = torch.randn(256, 1, 16)
drops = {}
for cap in [1.0, 1.5, 2.2]:
    _, _ = layer(x, topk=6, cap=cap, lbda=0.15, use_token_dropping=True)
    dr = layer.last_drop_ratio
    drops[cap] = dr
    print(f"  cap={cap} drop_ratio={dr:.4f}")
assert drops[1.0] > drops[1.5] > drops[2.2], "drop_ratio 应随 cap 增大而减小"
assert drops[1.0] > 0.0, "cap=1.0 应有丢弃"
ok += 1

# ---- 自适应蒸馏权重机制验证 ----
from marvis_moe import Education, Monitor
edu = Education(64)
mon = Monitor(64)
mon.util = torch.rand(64)
mon.steps = 1
model_sm = __import__('marvis_moe_v7', fromlist=['MarvisMoE']).MarvisMoE(cfg)
xb_t = torch.from_numpy(np.asarray(Xtr[:64], dtype=np.float32))
l_adapt, _ = edu.soft_loss(model_sm, mon, xb=xb_t, adaptive=True)
l_fixed, _ = edu.soft_loss(model_sm, mon, xb=xb_t, adaptive=False)
assert torch.isfinite(torch.as_tensor(l_adapt)), "adaptive loss 非有限"
print(f"  adaptive loss={l_adapt:.6f} fixed loss={l_fixed:.6f}")
ok += 1

print(f"\nALL SMOKE PASS ({ok} checks)")
