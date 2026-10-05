# -*- coding: utf-8 -*-
"""
Marvis MoE v7 —— 真民主原型：可学习元控制器（LearnedCouncil）
================================================================
v6 的治理是 if-else 阈值反馈（假民主）。v7 把议会投票升级为
REINFORCE 训练的可学习元控制器：输入训练状态，输出治理超参。

对比三方（同 seed / 同数据 / 同主模型）：
- fixed : 固定超参（lbda=0.15, cap=1.25, k=6, edu=0）——标准 MoE 近似
- rules : v6 if-else 治理（假民主）
- learned: LearnedCouncil + REINFORCE（真民主）

控制器设计：
- 状态 s = [util_cv, loss_rel, entropy_norm, grad_norm]（归一化到 ~[0,1]）
- 动作 a = [lbda, cap, k, edu]，sigmoid 映射到合理范围
- 奖励 r = 10*(ref_loss - cur_loss) - beta*util_cv - gamma*abs(k - base_k)
  即：压损失为主、压负载不均、约束 k 不飘远（避免靠增大计算量刷收益）
- 每 council_every 步采样动作，跑 G 步后用该段奖励做 REINFORCE 更新，
  baseline 用奖励滑动均值（减方差）

说明：层次1（专家协商）与层次2（多门控集成）暂不实现，需先解决
"协商不破坏稀疏性"的结构问题（候选集两级协商），留作 v8。
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marvis_moe import Config, MarvisMoE, Monitor, make_synthetic, set_seed


class LearnedCouncil(nn.Module):
    """可学习元控制器（议会）：状态 -> 治理超参。REINFORCE 训练。

    control_k=True ：动作 = [lbda, cap, k, edu]（完整控制，learned 模式）
    control_k=False：剥夺 k 控制权，动作 = [lbda, cap, edu]（learned_nok 模式）
    """
    def __init__(self, state_dim=4, hidden=32, E=64, topk=6, control_k=True):
        super().__init__()
        self.control_k = control_k
        self.n_actions = 4 if control_k else 3
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, self.n_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.n_actions) - 1.5)
        self.E = E
        self.topk = topk
        self.base_k = topk
        # 归一化参考
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))

    def forward(self, state, explore=True):
        s = (state - self.state_mean) / (self.state_std + 1e-6)
        mu = self.net(s)
        if explore and self.training:
            std = torch.exp(self.log_std).clamp(0.05, 1.0)
            a = mu + torch.randn_like(mu) * std
        else:
            a = mu
        log_prob = -0.5 * ((a - mu) / torch.exp(self.log_std).clamp(0.05, 1.0)) ** 2 \
                   - torch.log(torch.exp(self.log_std).clamp(0.05, 1.0)) - 0.5 * math.log(2 * math.pi)
        log_prob = log_prob.sum(-1)
        # 映射：lbda∈[0,0.5], cap∈[0.8,2.2], k∈[base_k±16]（clip 到 [2,E]）, edu∈[0,0.3]
        # k 用围绕 base_k 的窄区间：原 [2,E] 均匀映射使初始 k≈33，控制器靠大 k 刷
        # Δloss 收益，k 惩罚被淹没；±6 过窄导致 k 无动态（探索不足），扩到 ±16
        # 保持惩罚可传导的同时恢复动态（机制修正，见 CONFIG_NOTES）。
        # 动作值仅作训练超参使用（不参与本图梯度），detach 避免 float() 警告；
        # log_prob 保留梯度路径（REINFORCE 更新用）。
        lbda = torch.sigmoid(a[0]).detach() * 0.5
        cap = torch.sigmoid(a[1]).detach() * 1.4 + 0.8
        if self.control_k:
            k = int((torch.sigmoid(a[2]).detach() - 0.5) * 16) + self.base_k
            k = min(max(k, 2), self.E)
            edu = torch.sigmoid(a[3]).detach() * 0.3
        else:
            k = self.topk                          # learned_nok：剥夺 k 控制权
            edu = torch.sigmoid(a[2]).detach() * 0.3
        return dict(lbda=float(lbda), cap=float(cap), k=k, edu=float(edu), log_prob=log_prob)

    def update_norm(self, states):
        # 在线状态归一化参考（前若干步收集）
        states = torch.stack(states)
        self.state_mean.copy_(states.mean(0))
        self.state_std.copy_(states.std(0).clamp_min(1e-3))


def make_state(mon, loss_now, grad_norm, base_k):
    """组装控制器状态向量（归一化到 ~[0,1] 量级）。"""
    loss_ref = mon.loss_ref if mon.loss_ref is not None else loss_now
    ent = mon.entropy_ema if mon.entropy_ema is not None else 0.5
    return torch.tensor([
        min(mon.util_cv, 1.0),
        float(loss_now / (loss_ref + 1e-8)),          # 相对损失
        float(ent / math.log(max(2, mon.E))),         # 归一化熵
        float(min(grad_norm, 5.0) / 5.0),             # 梯度范数
    ], dtype=torch.float)


def train_v7(cfg, Xtr, ytr, Xva, yva, council_mode='learned', seed=2026,
             epochs=12, batch_size=256, lr=2e-3, council_every=20, G=4,
             beta=0.5, gamma=0.1, fixed_cap=None, use_token_dropping=False):
    """主循环：与 v6 fit 相同，但议会可切换为 LearnedCouncil（REINFORCE）。

    模式：fixed / rules / learned / learned_nok / learned_decoupled
    消融：learned_ablate_loss（奖励去 Δloss）/ learned_ablate_cv（奖励去 CV 惩罚）
         / learned_ablate_k（奖励去 k 惩罚）
    容量：learned_capdrop（固定 cap + 可导容量惩罚）/ learned_tokendrop（固定 cap + Token Dropping）
    蒸馏：learned_no_edu（无蒸馏）/ learned_fixed_edu（固定权重蒸馏）
         / learned_adaptive_edu（自适应权重蒸馏）
    """
    assert council_mode in (
        'fixed', 'rules', 'learned', 'learned_nok', 'learned_decoupled',
        'learned_ablate_loss', 'learned_ablate_cv', 'learned_ablate_k',
        'learned_capdrop', 'learned_tokendrop',
        'learned_no_edu', 'learned_fixed_edu', 'learned_adaptive_edu',
    ), f"未知 council_mode: {council_mode}"
    set_seed(seed)
    model = MarvisMoE(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    n_batch = max(1, n // batch_size)

    council = None
    dec_plan = None   # learned_decoupled 三阶段规划（fixed/rules 下恒为 None）
    if council_mode in ('learned', 'learned_nok', 'learned_decoupled',
                        'learned_ablate_loss', 'learned_ablate_cv', 'learned_ablate_k',
                        'learned_capdrop', 'learned_tokendrop',
                        'learned_no_edu', 'learned_fixed_edu', 'learned_adaptive_edu'):
        council = LearnedCouncil(state_dim=4, E=cfg.E, topk=cfg.topk,
                                 control_k=(council_mode != 'learned_nok'))
        council_opt = torch.optim.Adam(council.parameters(), lr=1e-3)
        # 段聚合式 REINFORCE：每个决策段（council_every 步）用同一动作，
        # 段结束用该段平均奖励更新一次，避免同一 log_prob 被 backward 多次。
        reward_baseline = None
        step_actions = None   # 当前生效的治理参数
        cur_lp = None         # 当前段的 log_prob（带梯度）
        cur_r_sum = 0.0
        cur_cnt = 0
        prev_seg_avg = None   # 上一段平均 loss（Delta Loss 奖励基准）
        cur_seg_ce_sum = 0.0  # 当前段 loss 累积
        cur_seg_cnt = 0
        last_ce = 1.0         # 供决策状态使用（初始占位）
        last_grad = 0.0
        collect_states = []
        # learned_decoupled 三阶段规划（阶段1 冻结Council / 阶段2 冻结主模型 / 阶段3 联合）
        if council_mode == 'learned_decoupled':
            if epochs >= 12:
                _s1, _s2, _s3 = 6, 6, epochs - 12
            else:
                _s1 = max(1, epochs // 3)
                _s2 = max(1, epochs // 3)
                _s3 = epochs - _s1 - _s2
            dec_plan = (_s1, _s2, _s3)
        else:
            dec_plan = None

    history = dict(loss=[], acc=[], val_acc=[], std=[], cv=[], events=[], rewards=[],
                   k_seq=[], council_log=[])

    for ep in range(epochs):
        # learned_decoupled 阶段判定
        freeze_council = False
        freeze_main = False
        if dec_plan is not None:
            s1, s2, s3 = dec_plan
            if ep < s1:
                freeze_council = True
            elif ep < s1 + s2:
                freeze_main = True
        if freeze_main:
            for p in model.parameters():
                p.requires_grad_(False)
        perm = torch.randperm(n)
        for bi in range(n_batch):
            idx = perm[bi * batch_size:(bi + 1) * batch_size]
            xb, yb = Xtr[idx], ytr[idx]

            # ---- 治理参数确定 ----
            if council_mode == 'fixed':
                params = dict(lbda=0.15, cap=1.25, k=cfg.topk, edu=0.0)
            elif council_mode == 'rules':
                # v6 假民主 if-else
                if model.mon.steps % council_every == 0:
                    model.council.step(model.mon)
                params = dict(lbda=model.council.lbda, cap=model.council.cap,
                              k=model.council.k, edu=model.council.edu)
            else:  # learned / learned_nok / learned_decoupled
                if step_actions is None or model.mon.steps % council_every == 0:
                    # 段结束：用上一段的平均奖励做一次 REINFORCE 更新
                    if cur_lp is not None and cur_cnt > 0:
                        r_mean = cur_r_sum / cur_cnt
                        if reward_baseline is None:
                            reward_baseline = r_mean
                        else:
                            reward_baseline = 0.9 * reward_baseline + 0.1 * r_mean
                        if freeze_council:
                            # 阶段1（冻结Council）：只更新 baseline，不反传参数
                            cur_lp = None
                        else:
                            adv = r_mean - reward_baseline
                            council_opt.zero_grad()
                            # 熵正则：鼓励探索、防 log_std 崩溃（红队#3 修复）
                            loss_rl = -(cur_lp * adv) - 0.01 * council.log_std.mean()
                            loss_rl.backward()
                            council_opt.step()
                            history['events'].append(('learned_council', model.mon.steps,
                                                      dict(lbda=params['lbda'], cap=params['cap'],
                                                           k=params['k'], edu=params['edu'])))
                    # 采样新动作：必须在有梯度模式下进行，log_prob 需要
                    # 梯度路径回传到 council 参数（REINFORCE 的核心）。
                    state = make_state(model.mon, last_ce, last_grad, cfg.topk)
                    step_actions = council(state, explore=not freeze_council)
                    collect_states.append(state.detach())
                    if len(collect_states) >= 10:   # 每10段更新一次状态归一化（红队#2 修复）
                        council.update_norm(collect_states)
                        collect_states = []
                    # 段切换：上一段平均 loss 作为 Delta Loss 奖励基准
                    if cur_seg_cnt > 0:
                        prev_seg_avg = cur_seg_ce_sum / cur_seg_cnt
                    cur_seg_ce_sum = 0.0
                    cur_seg_cnt = 0
                    cur_lp = step_actions['log_prob']
                    if freeze_council:
                        cur_lp = None          # 阶段1：不保留 REINFORCE 梯度路径
                    cur_r_sum = 0.0
                    cur_cnt = 0
                    # 控制器行为日志（2.4 可视化用）：每次采样记一段
                    history['k_seq'].append(step_actions['k'])
                    history['council_log'].append(dict(
                        step=model.mon.steps,
                        lbda=step_actions['lbda'], cap=step_actions['cap'],
                        k=step_actions['k'], edu=step_actions['edu'],
                        freeze_council=freeze_council, freeze_main=freeze_main))
                params = dict(lbda=step_actions['lbda'], cap=step_actions['cap'],
                              k=step_actions['k'], edu=step_actions['edu'])
                # 阶段 B：固定 cap（learned_capdrop/learned_tokendrop 只对比容量机制，
                # 控制器不再控制 cap 值，保证三档因子可比）
                if council_mode in ('learned_capdrop', 'learned_tokendrop'):
                    params['cap'] = fixed_cap if fixed_cap is not None else step_actions['cap']
                # 阶段 C：蒸馏对照（no_edu 关闭；fixed/adaptive 用固定 edu=0.15 保证蒸馏开启）
                if council_mode == 'learned_no_edu':
                    params['edu'] = 0.0
                elif council_mode in ('learned_fixed_edu', 'learned_adaptive_edu'):
                    params['edu'] = 0.15

            # ---- 前向/损失 ----
            opt.zero_grad()
            logits, aux = model(xb, use_token_dropping=use_token_dropping)
            ce = F.cross_entropy(logits, yb)
            total = ce + params['lbda'] * aux
            if council_mode in ('rules', 'learned', 'learned_nok', 'learned_decoupled',
                                'learned_ablate_loss', 'learned_ablate_cv', 'learned_ablate_k',
                                'learned_capdrop', 'learned_tokendrop',
                                'learned_no_edu', 'learned_fixed_edu', 'learned_adaptive_edu') \
                    and params['edu'] > 0 and model.mon.steps % 50 == 0:
                dloss, nw = model.education.soft_loss(model, model.mon, xb,
                                                      adaptive=(council_mode == 'learned_adaptive_edu'))
                total = total + params['edu'] * dloss
            if not freeze_main:
                total.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
                opt.step()
            else:
                # 阶段2（冻结主模型）：前向仅作监控，不反传主模型
                grad_norm = 0.0

            # ---- 监控 ----
            with torch.no_grad():
                acc = (logits.argmax(-1) == yb).float().mean().item()
                util = torch.zeros(cfg.E)
                ent_sum, n_layers = 0.0, len(model.layers)
                for layer in model.layers:
                    util.scatter_add_(0, layer.last_idx2.reshape(-1),
                                      torch.ones(layer.last_idx2.numel()))
                    p = layer.last_full_p
                    ent_sum += float(-(p * p.clamp_min(1e-9).log()).sum(-1).mean())
                entropy = ent_sum / max(1, n_layers)
                model.mon.update(util, ce.item(), entropy)

            history['loss'].append(ce.item())
            history['acc'].append(acc)
            last_ce = ce.item()       # 供 learned 决策状态使用
            last_grad = grad_norm

            # ---- 学习型议会：段奖励累积（段结束统一 REINFORCE 更新）----
            if council_mode in ('learned', 'learned_nok', 'learned_decoupled',
                                'learned_ablate_loss', 'learned_ablate_cv', 'learned_ablate_k',
                                'learned_capdrop', 'learned_tokendrop',
                                'learned_no_edu', 'learned_fixed_edu', 'learned_adaptive_edu') \
                    and step_actions is not None:
                with torch.no_grad():
                    # Delta Loss 奖励：直接奖励相对上一段平均 loss 的下降（红队#1 修复）
                    # 消融：分别剔除 Δloss / CV 惩罚 / k 惩罚项，评估三项贡献（阶段 A）
                    base = prev_seg_avg if prev_seg_avg is not None else ce.item()
                    r = -(ce.item() - base) \
                        - beta * model.mon.util_cv \
                        - gamma * abs(step_actions['k'] - cfg.topk)
                    if council_mode == 'learned_ablate_loss':
                        r = - beta * model.mon.util_cv - gamma * abs(step_actions['k'] - cfg.topk)
                    elif council_mode == 'learned_ablate_cv':
                        r = -(ce.item() - base) - gamma * abs(step_actions['k'] - cfg.topk)
                    elif council_mode == 'learned_ablate_k':
                        r = -(ce.item() - base) - beta * model.mon.util_cv
                cur_r_sum += r
                cur_cnt += 1
                cur_seg_ce_sum += ce.item()
                cur_seg_cnt += 1
                history['rewards'].append(r)

            history.setdefault('drop_ratios', []).append(
                float(sum(getattr(l, 'last_drop_ratio', 0.0) for l in model.layers)) / max(1, len(model.layers)))
            history['std'].append(model.mon.util_std)
            history['cv'].append(model.mon.util_cv)

        # 每 epoch 验证
        if freeze_main:
            for p in model.parameters():       # 阶段切换后恢复主模型可训练
                p.requires_grad_(True)
        model.eval()
        with torch.no_grad():
            lv, _ = model(Xva)
            va = (lv.argmax(-1) == yva).float().mean().item()
        model.train()
        history['val_acc'].append(va)

    history['final_util_std'] = model.mon.util_std
    history['final_cv'] = model.mon.util_cv
    history['final_active'] = model.mon.active_frac
    # 汇总指标（run_experiments CSV 用）
    history['train_acc'] = float(np.mean(history['acc'][-30:])) if history['acc'] else -1.0
    history['avg_loss'] = float(np.mean(history['loss'][-30:])) if history['loss'] else -1.0
    history['avg_k'] = float(np.mean(history['k_seq'])) if history['k_seq'] else float(cfg.topk)
    history['drop_ratio'] = float(np.mean(history['drop_ratios'])) if history.get('drop_ratios') else 0.0
    return model, history, council


def run_compare(seeds=(2026, 2027, 2028), epochs=12):
    d, E, S, L, K = 16, 64, 2, 2, 8
    cfg = Config(d=d, h=32, E=E, S=S, L=L, topk=6, out_dim=K)

    # 数据（每个 seed 独立生成，保证评价的是跨数据泛化）
    datasets = {}
    for sd in seeds:
        X, y = make_synthetic(n=4096, d=d, k=K, seed=sd)
        perm = torch.randperm(X.shape[0])
        X, y = X[perm], y[perm]
        datasets[sd] = (X[:3072], y[:3072], X[3072:], y[3072:])

    modes = ['fixed', 'rules', 'learned']
    results = {m: dict(va=[], cv=[], act=[], ev=0) for m in modes}

    for sd in seeds:
        Xtr, ytr, Xva, yva = datasets[sd]
        for m in modes:
            print(f"\n>>> seed={sd} mode={m}")
            model, h, _ = train_v7(cfg, Xtr, ytr, Xva, yva, council_mode=m, seed=sd, epochs=epochs)
            results[m]['va'].append(h['val_acc'][-1])
            results[m]['cv'].append(h['final_cv'])
            results[m]['act'].append(h['final_active'])
            results[m]['ev'] += len(h['events'])
            print(f"  验证acc={h['val_acc'][-1]:.4f} 负载CV={h['final_cv']:.4f} 活跃={h['final_active']:.2%} 事件={len(h['events'])}")

    print("\n" + "=" * 78)
    print("多 seed 汇总（均值 ± 标准差）")
    print("=" * 78)
    print(f"{'模式':<12}{'验证acc':>18}{'负载CV':>16}{'活跃':>12}{'治理事件':>10}")
    for m in modes:
        va = np.array(results[m]['va'])
        cv = np.array(results[m]['cv'])
        ac = np.array(results[m]['act'])
        print(f"{m:<12}{va.mean():.4f}±{va.std():.4f}    "
              f"{cv.mean():.4f}±{cv.std():.4f}    "
              f"{ac.mean():.2%}     {results[m]['ev']}")
    return results


def run_compare_mnist(seeds=(2026, 2027), epochs=12, n_train=12000, n_test=2000):
    """MNIST 上对比 fixed / rules / learned 三模式（真民主验证场）。"""
    from train_mnist import load_mnist, sample
    Xtr_all, ytr_all = load_mnist("train")
    Xte_all, yte_all = load_mnist("t10k")
    cfg = Config(d=784, h=256, E=64, S=2, L=2, topk=6, out_dim=10)

    datasets = {}
    for sd in seeds:
        Xtr_s, ytr_s = sample(Xtr_all, ytr_all, n_train, seed=sd)
        Xte_s, yte_s = sample(Xte_all, yte_all, n_test, seed=sd)
        perm = torch.randperm(n_train)
        va_n = n_train // 10
        datasets[sd] = (Xtr_s[perm[:-va_n]], ytr_s[perm[:-va_n]],
                        Xtr_s[perm[-va_n:]], ytr_s[perm[-va_n:]],
                        Xte_s, yte_s)

    modes = ['fixed', 'rules', 'learned']
    results = {m: dict(va=[], te=[], cv=[], act=[], ev=0) for m in modes}

    for sd in seeds:
        Xtr, ytr, Xva, yva, Xte, yte = datasets[sd]
        for m in modes:
            print(f"\n>>> seed={sd} mode={m} epochs={epochs}")
            model, h, _ = train_v7(cfg, Xtr, ytr, Xva, yva, council_mode=m,
                                   seed=sd, epochs=epochs, batch_size=256, lr=2e-3)
            model.eval()
            with torch.no_grad():
                logits, _ = model(Xte)
                te = (logits.argmax(-1) == yte).float().mean().item()
            results[m]['va'].append(h['val_acc'][-1])
            results[m]['te'].append(te)
            results[m]['cv'].append(h['final_cv'])
            results[m]['act'].append(h['final_active'])
            results[m]['ev'] += len(h['events'])
            print(f"  验证acc={h['val_acc'][-1]:.4f} 测试acc={te:.4f} "
                  f"负载CV={h['final_cv']:.4f} 活跃={h['final_active']:.2%} 事件={len(h['events'])}")

    print("\n" + "=" * 84)
    print("MNIST 多 seed 汇总（均值 ± 标准差）")
    print("=" * 84)
    print(f"{'模式':<12}{'验证acc':>18}{'测试acc':>16}{'负载CV':>16}{'活跃':>12}{'治理事件':>10}")
    for m in modes:
        va = np.array(results[m]['va'])
        te = np.array(results[m]['te'])
        cv = np.array(results[m]['cv'])
        ac = np.array(results[m]['act'])
        print(f"{m:<12}{va.mean():.4f}±{va.std():.4f}  "
              f"{te.mean():.4f}±{te.std():.4f}  "
              f"{cv.mean():.4f}±{cv.std():.4f}  "
              f"{ac.mean():.2%}     {results[m]['ev']}")
    return results


if __name__ == '__main__':
    run_compare()
