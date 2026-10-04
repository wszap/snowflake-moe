# -*- coding: utf-8 -*-
"""TDD：LearnedCouncil + 段聚合 REINFORCE 的单元测试（先测试后改码）。

覆盖：
1. 动作输出在合法范围内（lbda∈[0,0.5], cap∈[0.8,2.2], k∈[2,E], edu∈[0,0.3]）且非 NaN
2. 离散化正确（int 截断的 k）
3. log_prob 带梯度路径，backward 成功（元梯度可训练）
4. make_state 归一化状态形状/有限性
5. 段聚合 REINFORCE 更新：真实小规模训练循环不报错、有奖励、事件可记录

运行（须先保证 output 与 temp 在 sys.path）：
    python test_learned_council.py
"""
import math
import sys

import torch

from marvis_moe import Config, MarvisMoE, Monitor, make_synthetic, set_seed
from marvis_moe_v7 import LearnedCouncil, make_state, train_v7


def test_action_bounds_and_nan():
    set_seed(0)
    c = LearnedCouncil(state_dim=4, hidden=32, E=64, topk=6)
    s = torch.tensor([0.5, 1.0, 0.5, 0.1])
    for explore in (False, True):
        a = c(s, explore=explore)
        assert not math.isnan(a['lbda']) and not math.isnan(a['cap'])
        assert not math.isnan(a['edu'])
        assert 0.0 <= a['lbda'] <= 0.5 + 1e-6
        assert 0.8 <= a['cap'] <= 2.2 + 1e-6
        assert 2 <= a['k'] <= 64
        assert 0.0 <= a['edu'] <= 0.3 + 1e-6
        assert torch.isfinite(a['log_prob'])
    print("[PASS] test_action_bounds_and_nan")


def test_k_discretization():
    c = LearnedCouncil(state_dim=4, hidden=32, E=64, topk=6)
    # 清零网络权重，使 a[2]=0（纯测试离散化映射，不依赖随机初始化）
    for p in c.net.parameters():
        p.data.zero_()
    c.eval()
    s = torch.tensor([0.0, 0.0, 0.0, 0.0])
    a = c(s, explore=False)
    # sigmoid(0)=0.5 -> k = int(0)+base_k = 6（k 动作空间改为围绕 base_k 的窄区间）
    assert a['k'] == 6, a['k']
    print("[PASS] test_k_discretization (k=%d)" % a['k'])


def test_log_prob_grad():
    c = LearnedCouncil(state_dim=4, hidden=32, E=64, topk=6)
    s = torch.tensor([0.5, 1.0, 0.5, 0.1])
    a = c(s, explore=True)
    lp = a['log_prob']
    assert lp.requires_grad
    lp.backward()
    assert c.net[0].weight.grad is not None
    assert c.log_std.grad is not None
    print("[PASS] test_log_prob_grad")


def test_make_state():
    mon = Monitor(8)
    mon.loss_ref = 1.0
    mon.loss_ema = 1.05
    mon.entropy_ema = 0.6
    mon.util = torch.ones(8) * 2.0          # cv = 0
    s = make_state(mon, 1.2, 0.3, 6)
    assert s.shape == (4,)
    assert torch.isfinite(s).all()
    assert s[1] > 1.0                        # loss 相对值
    print("[PASS] test_make_state", [round(float(v), 4) for v in s])


def test_reinforce_segment_update():
    d, E, S, L, K = 16, 16, 2, 1, 8
    cfg = Config(d=d, h=16, E=E, S=S, L=L, topk=4, out_dim=K)
    X, y = make_synthetic(n=512, d=d, k=K, seed=1)
    perm = torch.randperm(512)
    Xtr, ytr, Xva, yva = X[perm[:384]], y[perm[:384]], X[perm[384:]], y[perm[384:]]
    # council_every=2：steps 0,1,2,3 内 steps=0 决策、steps=2 决策并提交段1
    model, h, council = train_v7(cfg, Xtr, ytr, Xva, yva, council_mode='learned',
                                 seed=1, epochs=1, batch_size=128, council_every=2)
    assert len(h['rewards']) >= 3           # 3 batch 每批都有奖励
    assert len(h['events']) >= 1            # 至少完成一次段更新
    print("[PASS] test_reinforce_segment_update (events=%d)" % len(h['events']))


def test_learned_nok_k_fixed():
    """learned_nok：剥夺 k 控制权后，k 恒等于 topk（动作空间 3 维）。"""
    c = LearnedCouncil(state_dim=4, hidden=32, E=64, topk=6, control_k=False)
    c.eval()
    for p in c.net.parameters():
        p.data.zero_()
    s = torch.zeros(4)
    a = c(s, explore=False)
    assert a['k'] == 6, a['k']
    assert 0.0 <= a['lbda'] <= 0.5 + 1e-6
    assert 0.8 <= a['cap'] <= 2.2 + 1e-6
    assert 0.0 <= a['edu'] <= 0.3 + 1e-6
    assert torch.isfinite(a['log_prob'])
    print("[PASS] test_learned_nok_k_fixed (k=%d)" % a['k'])


def test_learned_nok_train_smoke():
    """learned_nok 训练冒烟：不报错、有奖励、avg_k 恒为 topk。"""
    d, E, S, L, K = 16, 16, 2, 1, 8
    cfg = Config(d=d, h=16, E=E, S=S, L=L, topk=4, out_dim=K)
    X, y = make_synthetic(n=512, d=d, k=K, seed=1)
    perm = torch.randperm(512)
    Xtr, ytr, Xva, yva = X[perm[:384]], y[perm[:384]], X[perm[384:]], y[perm[384:]]
    model, h, council = train_v7(cfg, Xtr, ytr, Xva, yva, council_mode='learned_nok',
                                 seed=1, epochs=1, batch_size=128, council_every=2)
    assert h['avg_k'] == 4                    # k 恒为 topk
    assert len(h['rewards']) >= 3
    assert len(h['k_seq']) == len(h['council_log']) and len(h['k_seq']) > 0
    print("[PASS] test_learned_nok_train_smoke (rewards=%d)" % len(h['rewards']))


def test_learned_decoupled_smoke():
    """learned_decoupled 三阶段冒烟：8 epoch 覆盖冻结Council/冻结主模型/联合三阶段，不报错。"""
    d, E, S, L, K = 16, 16, 2, 1, 8
    cfg = Config(d=d, h=16, E=E, S=S, L=L, topk=4, out_dim=K)
    X, y = make_synthetic(n=512, d=d, k=K, seed=2)
    perm = torch.randperm(512)
    Xtr, ytr, Xva, yva = X[perm[:384]], y[perm[:384]], X[perm[384:]], y[perm[384:]]
    model, h, council = train_v7(cfg, Xtr, ytr, Xva, yva, council_mode='learned_decoupled',
                                 seed=2, epochs=8, batch_size=128, council_every=2)
    assert len(h['val_acc']) == 8
    assert torch.isfinite(torch.tensor(h['loss'][-1]))
    assert torch.isfinite(torch.tensor(h['final_cv']))
    print("[PASS] test_learned_decoupled_smoke (loss=%.4f cv=%.4f)" % (h['loss'][-1], h['final_cv']))


if __name__ == '__main__':
    test_action_bounds_and_nan()
    test_k_discretization()
    test_log_prob_grad()
    test_make_state()
    test_reinforce_segment_update()
    test_learned_nok_k_fixed()
    test_learned_nok_train_smoke()
    test_learned_decoupled_smoke()
    print("\nALL TESTS PASSED")
