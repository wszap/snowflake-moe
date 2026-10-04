# -*- coding: utf-8 -*-
"""
Marvis MoE —— 用户构想模型框架（v5，真实可训练版）
=====================================================
把「融合制国家治理模拟」的取舍结论落地为真实 AI 训练机制，
结构组件对齐 DeepSeek 类 MoE（细粒度路由专家 + 共享专家 + 路由器
+ 容量因子软约束 + aux 负载均衡 + 动态 Top-K）。

关键约定
--------
1. 智能体分工数量不设限：专家总数 E、共享专家数 S、层数 L 全部参数化，
   框架在任意规模下成立（演示用小规模，放大只需改 config）。
2. 制度模拟仅为取舍参考，不是最终架构：
   - 市场部       -> 可学习路由器门控打分
   - 计划委       -> 每层共享专家保底激活（DeepSeek 同款设计）
   - 联邦政府     -> 负载漂移检测 + 紧急扩容（动态容量因子 / 动态 Top-K）
   - 议会         -> 治理委员会投票调参（民主集中制，实践-认识-再实践）
   - 税务局       -> auxiliary balance loss + 容量因子 bias
   - 公益组织     -> 训练监控（利用率 / 损失 EMA / 熵），生成治理依据
   - 教育部       -> 弱专家向强专家知识蒸馏（量变质变 + 否定之否定）
   - 特区港       -> 探索型专家（高噪声 + 探索偏置，保多样性）
   - 气象局       -> 容量需求预测（利用率 EMA 外推）
   - 总统         -> 多门控集成（路由 logits 与治理调节信号融合）
3. 运行环境：纯 PyTorch CPU，自造合成数据即可验证，无需外部数据集。
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(s=2026):
    torch.manual_seed(s)
    np.random.seed(s)


# ================================================================ 配置
class Config:
    def __init__(self, d=16, h=32, E=64, S=2, L=2, topk=6,
                 special_bias=0.2, expert_noise=0.02, cap=1.25,
                 lbda=0.15, out_dim=None):
        self.d = d            # 输入/隐藏维度
        self.h = h            # 专家内层宽度
        self.E = E            # 路由专家总数（智能体数量，不设限）
        self.S = S            # 共享专家数（计划保底）
        self.L = L            # MoE 层数
        self.topk = topk      # 每 token 激活专家数
        self.special_bias = special_bias   # 特区探索偏置
        self.expert_noise = expert_noise   # 专家噪声（特区更大）
        self.cap = cap        # 容量因子
        self.lbda = lbda      # aux 平衡损失权重
        self.out_dim = out_dim if out_dim else d


# ================================================================ 基础组件
class SwiGLU(nn.Module):
    """类 DeepSeek 的 SwiGLU 专家内层。"""
    def __init__(self, d, h):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Expert(nn.Module):
    """单个细粒度专家。special=True 时是『特区』探索型专家（高噪声）。"""
    def __init__(self, d, h, noise=0.0, special=False):
        super().__init__()
        self.net = SwiGLU(d, h)
        self.noise = noise
        self.special = special

    def forward(self, x):
        z = self.net(x)
        if self.training and self.noise > 0:
            z = z + torch.randn_like(z) * self.noise
        return z


class Router(nn.Module):
    """门控打分：市场部产出 logits，总统做多信号融合（含治理调节）。"""
    def __init__(self, d, E, special_bias=0.0):
        super().__init__()
        self.gate = nn.Linear(d, E, bias=False)   # 市场部打分
        self.special_bias = special_bias          # 特区探索偏置

    def score(self, x, adjust=None):
        logits = self.gate(x)
        if adjust is not None:                    # 联邦/议会传入的调节信号
            logits = logits + adjust.view(1, -1)
        if self.special_bias > 0 and self.training:
            n_spec = max(1, int(logits.size(-1) * 0.05))
            logits[..., :n_spec] = logits[..., :n_spec] + self.special_bias
        return logits


# ================================================================ 单层 MoE
class MoELayer(nn.Module):
    """细粒度路由专家 + 共享专家 + 容量因子软约束 + aux loss。"""
    def __init__(self, cfg):
        super().__init__()
        d, h, E, S = cfg.d, cfg.h, cfg.E, cfg.S
        self.d, self.E, self.S = d, E, S
        self.router = Router(d, E, special_bias=cfg.special_bias)
        n_spec = max(1, int(E * 0.05))
        self.experts = nn.ModuleList([
            Expert(d, h, noise=cfg.expert_noise if i >= n_spec else cfg.expert_noise * 3,
                   special=(i < n_spec))
            for i in range(E)])
        self.shared = nn.ModuleList([Expert(d, h, noise=0.0) for _ in range(S)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, topk=None, cap=None, lbda=None, adjust=None):
        B, T, d = x.shape
        x = x.reshape(B * T, d)
        routed = self.router.score(self.norm(x), adjust=adjust)

        # 动态 Top-K：打分方差小（拿不准）时多激活（联邦紧急扩容）
        k = topk if topk is not None else 6
        var = routed.var(dim=-1).mean().item()
        if var < 0.15:
            k = min(k + 1, self.E)

        # 容量因子软约束（税务局：超容专家 logits 惩罚）
        cap = cap if cap is not None else 1.25
        lbda = lbda if lbda is not None else 0.15
        topk_logits, idx = torch.topk(routed, k, dim=-1)
        probs = F.softmax(topk_logits, dim=-1)
        ones = torch.ones_like(routed)
        count = torch.zeros(self.E, device=x.device)
        count.scatter_add_(0, idx.reshape(-1), ones.reshape(-1))
        cap_count = math.ceil(x.shape[0] / self.E * cap)
        over = torch.clamp(count - cap_count, min=0.0)
        routed = routed - over.view(1, -1) * (0.5 + lbda)

        topk_logits2, idx2 = torch.topk(routed, k, dim=-1)
        probs2 = F.softmax(topk_logits2, dim=-1)

        # 聚合路由专家输出（细粒度）：index_add 累加每个 token 被多专家激活的加权和
        out = torch.zeros_like(x)
        flat_idx = idx2.reshape(-1)
        flat_probs = probs2.reshape(-1)
        tok_ids = torch.arange(x.shape[0], device=x.device).repeat_interleave(k)
        for e in range(self.E):
            m = flat_idx == e
            if m.any():
                src = flat_probs[m].unsqueeze(-1) * self.experts[e](x[tok_ids[m]])
                out.index_add_(0, tok_ids[m], src)

        # 共享专家保底（计划委：人人可用，稳定收敛）
        for s in self.shared:
            out = out + s(x)

        # 负载均衡 aux loss（税务局：频率 x 平均概率，越小越均衡）
        full_p = torch.zeros(B * T, self.E, device=x.device)
        full_p.scatter_(1, idx2, probs2)
        freq = count / count.sum().clamp_min(1.0)
        mean_p = full_p.mean(dim=0)
        aux = (freq * mean_p).sum() * self.E

        out = out.reshape(B, T, d)
        return out, aux


# ================================================================ 治理监控（公益组织）
class Monitor:
    """记录利用率、熵、损失 EMA，作为议会/联邦/教育部的决策依据。"""
    def __init__(self, E):
        self.E = E
        self.util = torch.zeros(E)
        self.steps = 0
        self.loss_ema = None
        self.entropy_ema = None

    def update(self, util, loss, entropy):
        self.util = 0.9 * self.util + 0.1 * util.cpu().float()
        self.steps += 1
        w = 0.1
        self.loss_ema = loss if self.loss_ema is None else (1 - w) * self.loss_ema + w * loss
        self.entropy_ema = entropy if self.entropy_ema is None else (1 - w) * self.entropy_ema + w * entropy

    @property
    def util_std(self):
        if self.steps < 2:
            return 0.0
        return float(self.util.std().item())

    @property
    def active_frac(self):
        return float((self.util > 0.01).float().mean().item())


# ================================================================ 治理循环（议会/联邦/教育部/气象局）
class Council:
    """民主集中制：根据监控指标投票调整训练超参。"""
    def __init__(self, cfg):
        self.lbda = cfg.lbda
        self.cap = cfg.cap
        self.edu = 0.0
        self.k = cfg.topk
        self.loss_up_streak = 0

    def step(self, mon: Monitor):
        # 议会投票（规则投票：均衡差 → 加平衡税；损失升 → 扩容+教育）
        if mon.util_std > 0.30:
            self.lbda = min(0.5, self.lbda + 0.05)
        else:
            self.lbda = max(0.02, self.lbda - 0.02)

        if mon.loss_ema is not None:
            self.loss_up_streak = self.loss_up_streak + 1 if mon.loss_ema > 1.5 else 0

        if self.loss_up_streak >= 3:          # 联邦紧急调配（量变质变触发）
            self.cap = min(2.0, self.cap + 0.25)
            self.edu = min(0.3, self.edu + 0.05)
        else:
            self.cap = max(1.0, self.cap - 0.01)

        # 气象局：负载熵低（太确定）时减少激活；熵高时扩容
        if mon.entropy_ema is not None:
            self.k = self.k + 1 if mon.entropy_ema < 0.5 else max(2, self.k - 1)

        return dict(lbda=self.lbda, cap=self.cap, edu=self.edu, k=self.k)


class Education:
    """教育部：弱专家向强专家蒸馏（量变质变累积 → 否定之否定螺旋升级）。"""
    def __init__(self, E, weak_frac=0.1, t=0.05):
        self.E = E
        self.weak_frac = weak_frac
        self.t = t

    def apply(self, layer: MoELayer, mon: Monitor):
        util = mon.util
        if util.sum().item() < 1e-9:
            return 0
        n_weak = max(1, int(self.E * self.weak_frac))
        order = torch.argsort(util)
        weak_idx = order[:n_weak].tolist()
        strong_idx = order[-n_weak:].tolist()
        for wi, si in zip(weak_idx, strong_idx):
            we, se = layer.experts[wi], layer.experts[si]
            for (wn, wp), (sn, sp) in zip(we.named_parameters(), se.named_parameters()):
                if wn == sn and wp.shape == sp.shape:
                    wp.data = (1 - self.t) * wp.data + self.t * sp.data
                    if layer.experts[wi].special:
                        wp.data = wp.data + torch.randn_like(wp.data) * 0.01  # 特区保留探索
        return n_weak


# ================================================================ 主体模型
class MarvisMoE(nn.Module):
    """可缩放 MoE 框架：E 专家 × L 层 + 共享专家 + 治理循环。"""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.layers = nn.ModuleList([MoELayer(cfg) for _ in range(cfg.L)])
        self.head = nn.Linear(cfg.d, cfg.out_dim)
        self.council = Council(cfg)
        self.education = Education(cfg.E)
        self.mon = Monitor(cfg.E)

    def forward(self, x, adjust=None):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.in_proj(x)
        aux_total = 0.0
        for i, layer in enumerate(self.layers):
            x, aux = layer(x, topk=self.council.k, cap=self.council.cap,
                           lbda=self.council.lbda, adjust=adjust)
            aux_total = aux_total + aux
        x = x.mean(dim=1)
        return self.head(x), aux_total

    # ---------------- 训练循环（实践-认识-再实践） ----------------
    def fit(self, X, y, epochs=6, batch_size=256, lr=2e-3, gov=True,
            council_every=20, edu_every=50, seed=2026):
        set_seed(seed)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n = X.shape[0]
        n_batch = max(1, n // batch_size)
        history = dict(loss=[], acc=[], std=[], aux=[], events=[])

        for ep in range(epochs):
            perm = torch.randperm(n)
            for bi in range(n_batch):
                idx = perm[bi * batch_size:(bi + 1) * batch_size]
                xb, yb = X[idx], y[idx]
                opt.zero_grad()
                logits, aux = self(xb)
                ce = F.cross_entropy(logits, yb)
                total = ce + self.council.lbda * aux
                total.backward()
                opt.step()

                with torch.no_grad():
                    acc = (logits.argmax(-1) == yb).float().mean().item()
                    # 利用率统计：统计各层第一个路由层选中次数
                    layer = self.layers[0]
                    routed = layer.router.score(layer.norm(xb.unsqueeze(1).reshape(-1, self.cfg.d)))
                    kk = self.council.k
                    _, idxk = torch.topk(routed, kk, dim=-1)
                    util = torch.zeros(self.cfg.E)
                    util.scatter_add_(0, idxk.reshape(-1), torch.ones(idxk.numel()))
                    entropy = float(-(F.softmax(routed, -1) * F.log_softmax(routed, -1)).sum(-1).mean().item())
                    self.mon.update(util, ce.item(), entropy)

                history['loss'].append(ce.item())
                history['acc'].append(acc)
                history['aux'].append(aux.item())

                # 治理循环：每 council_every 步议会投票（民主集中制）
                if gov and self.mon.steps % council_every == 0:
                    params = self.council.step(self.mon)
                    history['events'].append(('council', self.mon.steps, params.copy()))
                # 教育部：弱专家蒸馏（否定之否定）
                if gov and self.mon.steps % edu_every == 0:
                    nw = self.education.apply(self.layers[0], self.mon)
                    history['events'].append(('educate', self.mon.steps, {'weak': nw}))

                history['std'].append(self.mon.util_std)

        # 返回监控快照
        history['final_util_std'] = self.mon.util_std
        history['final_active'] = self.mon.active_frac
        return history


# ================================================================ 合成数据（无外部依赖）
def make_synthetic(n=4096, d=16, k=8, seed=2026):
    """生成 k 类非线性分类任务：类别中心在超球面随机方向 + 类间重叠噪声。"""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * 3.0
    X = []
    y = []
    per = n // k
    for c in range(k):
        # 每个类中心加噪声，并用其它类中心做小幅干扰制造类间重叠（非线性）
        base = centers[c][None, :] + rng.standard_normal((per, d)) * 0.8
        mix = centers[(c + 1) % k][None, :] + rng.standard_normal((per, d)) * 0.8
        w = rng.uniform(0.2, 0.4, (per, 1))
        X.append(base * (1 - w) + mix * w)
        y.append(np.full(per, c))
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return torch.from_numpy(X), torch.from_numpy(y)


# ================================================================ 演示入口
def main():
    set_seed(2026)
    print("=" * 70)
    print("Marvis MoE v5：真实可训练框架（E 路由专家 + S 共享专家 + 治理循环）")
    print("=" * 70)

    d, E, S, L, K = 16, 64, 2, 2, 8
    cfg = Config(d=d, h=32, E=E, S=S, L=L, topk=6, out_dim=K)
    X, y = make_synthetic(n=4096, d=d, k=K, seed=2026)

    # 公平对比：同数据同种子，仅治理开关不同
    results = {}
    for gov in (False, True):
        set_seed(2026)
        model = MarvisMoE(cfg)
        print(f"\n>>> 训练中：治理={'开' if gov else '关'}  (E={E}, S={S}, L={L}, topk={model.council.k})")
        h = model.fit(X, y, epochs=6, batch_size=256, lr=2e-3, gov=gov,
                      council_every=20, edu_every=50)
        last = 50
        print(f"  正确率   : {np.mean(h['acc'][-last:]):.4f}")
        print(f"  末段损失 : {np.mean(h['loss'][-last:]):.4f}")
        print(f"  负载 std : {h['final_util_std']:.4f}")
        print(f"  活跃专家 : {h['final_active']:.2%}")
        if gov:
            print(f"  治理事件 : {len(h['events'])} 次 (council={sum(1 for e in h['events'] if e[0]=='council')}, educate={sum(1 for e in h['events'] if e[0]=='educate')})")
            final = h['events'][-1][2] if h['events'] else {}
            print(f"  末次治理参: λ_bal={model.council.lbda:.3f} cap={model.council.cap:.3f} edu={model.council.edu:.3f} k={model.council.k}")
        results[gov] = h

    print("\n" + "=" * 70)
    print("治理开/关公平对比（同 seed=2026，同任务流）")
    print("=" * 70)
    print(f"{'指标':<12}{'治理关':>14}{'治理开':>14}")
    for name, f in [('正确率', lambda h: np.mean(h['acc'][-50:])),
                    ('末段损失', lambda h: np.mean(h['loss'][-50:])),
                    ('负载std', lambda h: h['final_util_std']),
                    ('活跃专家比', lambda h: h['final_active'])]:
        print(f"{name:<12}{f(results[False]):>14.4f}{f(results[True]):>14.4f}")


if __name__ == '__main__':
    main()
