# -*- coding: utf-8 -*-
"""
Marvis MoE —— 用户构想模型框架（v6，审查修复版）
=====================================================
在 v5 基础上按审查意见修复：
- P0: count.scatter_add_ 改为 src 与 index 同形（ones_like(idx)），消除跨版本隐患
- P0: 专家聚合改为按专家排序分组（O(N log N)），去除每专家全量 mask 比较
- P0: 修正气象局熵与 k 的反向逻辑（熵低 -> 减 k，熵高 -> 扩容），并移除 MoELayer
      内与 Council 叠加的方差硬编码判据，动态 Top-K 由议会单点调度
- P1: 容量约束补可导容量惩罚 loss（期望占用超容部分进梯度），前向硬修正保留
- P1: aux loss 采用 DeepSeek 官方形式：freq 为 stop-grad 计数、mean_p 可导
- P2: Education 改为软蒸馏 MSE loss（不动参数，保留 Adam 状态与专家多样性，跳过特区）
- P2: Monitor 改为统计所有层、且使用实际路由决策（含容量修正后的 idx2 / 分布）
- P2: 治理阈值自适应（变异系数判据、相对损失上升判据、归一化熵），减少硬编码
- P3: 数据拆分训练/验证集，报告验证集正确率；新增同参数量密集 MLP baseline

组件映射（融合制国家治理 -> MoE）：
- 市场部 -> 可学习路由器门控打分
- 计划委 -> 每层共享专家保底激活（DeepSeek 同款直接相加设计）
- 联邦政府 -> 负载漂移检测 + 紧急扩容（动态容量因子 / 动态 Top-K）
- 议会 -> 治理委员会投票调参（民主集中制，实践-认识-再实践）
- 税务局 -> auxiliary balance loss + 可导容量惩罚
- 公益组织 -> 训练监控（利用率 / 损失 EMA / 熵），生成治理依据
- 教育部 -> 弱专家向强专家软蒸馏（量变质变 + 否定之否定，loss 形式不破坏参数）
- 特区港 -> 探索型专家（高噪声 + 探索偏置，保多样性，不参与蒸馏）
- 气象局 -> 路由熵预测（熵低减激活、熵高扩容）
- 总统 -> 多门控集成（路由 logits 与治理调节信号融合）

运行环境：纯 PyTorch CPU，自造合成数据即可验证，无需外部数据集。
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
                 lbda=0.15, cap_w=0.1, out_dim=None):
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
        self.cap_w = cap_w    # 可导容量惩罚权重
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
    """细粒度路由专家 + 共享专家 + 容量因子软约束 + aux loss（v6 修复版）。"""
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
        # Monitor 统计用：记录本层最近一次前向的实际路由决策（含容量修正后）
        self.last_idx2 = None
        self.last_full_p = None

    def forward(self, x, topk=None, cap=None, lbda=None, adjust=None):
        B, T, d = x.shape
        x = x.reshape(B * T, d)
        routed = self.router.score(self.norm(x), adjust=adjust)

        # Top-K：k 由议会统一调度（Council.step 依归一化熵动态调整），本层不再
        # 自调，避免"层内方差判据 + 议会熵判据"两处叠加导致 k 震荡
        k = topk if topk is not None else 6

        # 容量因子软约束（税务局：超容专家 logits 惩罚）
        # 前向硬修正：保证本 batch 内实际不超载（运行时行为，不产生梯度）
        cap = cap if cap is not None else 1.25
        lbda = lbda if lbda is not None else 0.15
        topk_logits, idx = torch.topk(routed, k, dim=-1)
        probs = F.softmax(topk_logits, dim=-1)
        # 计数：src 与 index 必须同形（ones_like(idx)），
        # 不依赖"src 比 index 长时前缀截断"的非标准行为
        count = torch.zeros(self.E, device=x.device)
        count.scatter_add_(0, idx.reshape(-1),
                           torch.ones_like(idx, dtype=torch.float).reshape(-1))
        cap_count = math.ceil(x.shape[0] / self.E * cap)
        over = torch.clamp(count - cap_count, min=0.0)
        routed = routed - over.view(1, -1) * (0.5 + lbda)

        topk_logits2, idx2 = torch.topk(routed, k, dim=-1)
        probs2 = F.softmax(topk_logits2, dim=-1)

        # 聚合路由专家输出：按专家排序 token 后分组前向
        # （总开销 O(N log N) + 各专家一次 kernel，替代每专家全量 mask 的 O(N*E)；
        #  GPU 放大时替换为 grouped_mm / DeepSpeed MoE 内核即可）
        out = torch.zeros_like(x)
        flat_idx = idx2.reshape(-1)
        flat_probs = probs2.reshape(-1)
        tok_ids = torch.arange(x.shape[0], device=x.device).repeat_interleave(k)
        order = torch.argsort(flat_idx, stable=True)
        s_idx, s_tok, s_prob = flat_idx[order], tok_ids[order], flat_probs[order]
        bounds = torch.searchsorted(s_idx, torch.arange(self.E, device=x.device) + 1)
        start = 0
        for e in range(self.E):
            end = int(bounds[e])
            if end > start:
                seg = slice(start, end)
                src = s_prob[seg].unsqueeze(-1) * self.experts[e](x[s_tok[seg]])
                out.index_add_(0, s_tok[seg], src)
            start = end

        # 共享专家保底（计划委：人人可用，稳定收敛；DeepSeek 同款直接相加）
        for s in self.shared:
            out = out + s(x)

        # 实际路由分布（容量修正后）：供 aux / 容量惩罚 / 监控统一使用
        full_p = torch.zeros(B * T, self.E, device=x.device)
        full_p.scatter_(1, idx2, probs2)

        # 标准 DeepSeek aux loss：freq(stop-grad 计数) x mean_p(可导概率)，越小越均衡。
        # 与官方实现一致，freq 本就视为常数（argmax 选择不可导）；
        # 梯度路径由 mean_p（经 full_p -> probs2）完整提供。
        freq = count / count.sum().clamp_min(1.0)
        mean_p = full_p.mean(dim=0)
        aux = (freq * mean_p).sum() * self.E

        # 可导容量惩罚：期望占用率 = sum(probs)/N（可导），超过平均激活率 k/E 的
        # 专家被惩罚（前向硬修正不产生梯度，由本项补上"避免超载"的梯度信号；
        # 阈值取 k/E 而非 cap/E：每 token 选 k 个专家时平均占用本就是 k/E，
        # 用 cap/E 会把几乎所有专家判为超容，惩罚尺度失控并压过分类 loss）。
        exp_occ = full_p.sum(dim=0) / x.shape[0]
        cap_loss = F.relu(exp_occ - k / self.E).sum()

        # 供 Monitor 统计：记录实际路由决策（含容量修正与议会 k）
        self.last_idx2 = idx2.detach()
        self.last_full_p = full_p.detach()

        out = out.reshape(B, T, d)
        return out, aux, cap_loss


# ================================================================ 治理监控（公益组织）
class Monitor:
    """记录利用率、熵、损失 EMA，作为议会/联邦/教育部的决策依据（自适应参考）。"""
    def __init__(self, E):
        self.E = E
        self.util = torch.zeros(E)
        self.steps = 0
        self.loss_ema = None
        self.loss_ref = None
        self.entropy_ema = None

    def update(self, util, loss, entropy):
        self.util = 0.9 * self.util + 0.1 * util.cpu().float()
        self.steps += 1
        w = 0.1
        self.loss_ema = loss if self.loss_ema is None else (1 - w) * self.loss_ema + w * loss
        self.loss_ref = loss if self.loss_ref is None else (1 - 0.05) * self.loss_ref + 0.05 * loss
        self.entropy_ema = entropy if self.entropy_ema is None else (1 - w) * self.entropy_ema + w * entropy

    @property
    def util_std(self):
        if self.steps < 2:
            return 0.0
        return float(self.util.std().item())

    @property
    def util_cv(self):
        """变异系数：负载不均衡度 / 平均负载，自适应 E 尺度（替代绝对 std 硬阈值）。"""
        m = float(self.util.mean().item())
        if m < 1e-9:
            return 0.0
        return float(self.util.std().item()) / m

    @property
    def active_frac(self):
        return float((self.util > 0.01).float().mean().item())


# ================================================================ 治理循环（议会/联邦/教育部/气象局）
class Council:
    """民主集中制：根据监控指标投票调整训练超参（v6：阈值自适应）。"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.lbda = cfg.lbda
        self.cap = cfg.cap
        self.edu = 0.0
        self.k = cfg.topk
        self.loss_up_streak = 0

    def step(self, mon: Monitor):
        # 议会投票：负载均衡用变异系数判据（相对量，自适应 E 尺度）
        if mon.util_cv > 0.5:
            self.lbda = min(0.5, self.lbda + 0.05)
        else:
            self.lbda = max(0.02, self.lbda - 0.02)

        # 联邦紧急调配：损失相对上升判据（loss_ema > 1.02 * 运行均值参考，
        # 替代硬编码绝对阈值 1.5）
        if mon.loss_ref is not None and mon.loss_ema > mon.loss_ref * 1.02:
            self.loss_up_streak += 1
        else:
            self.loss_up_streak = 0

        if self.loss_up_streak >= 3:          # 量变质变触发
            self.cap = min(2.0, self.cap + 0.25)
            self.edu = min(0.3, self.edu + 0.05)
        else:
            self.cap = max(1.0, self.cap - 0.01)

        # 气象局：归一化路由熵判据（熵低=太确定 -> 减少激活；熵高=不确定 -> 扩容）
        if mon.entropy_ema is not None:
            ent_norm = mon.entropy_ema / math.log(max(2, self.cfg.E))
            if ent_norm > 0.8:
                self.k = min(self.cfg.E, self.k + 1)
            elif ent_norm < 0.35:
                self.k = max(2, self.k - 1)

        return dict(lbda=self.lbda, cap=self.cap, edu=self.edu, k=self.k)


class Education:
    """教育部：弱专家向强专家软蒸馏（量变质变累积 -> 否定之否定螺旋升级）。
    v6 改为 loss 形式：不动参数、不破坏优化器状态、保留专家多样性、跳过特区。"""
    def __init__(self, E, weak_frac=0.1):
        self.E = E
        self.weak_frac = weak_frac

    def soft_loss(self, model, mon, xb):
        """返回 (蒸馏 loss, 弱专家数量)。弱专家输出向强专家输出对齐（MSE）。"""
        if mon.util.sum().item() < 1e-9:
            return 0.0, 0
        n_weak = max(1, int(self.E * self.weak_frac))
        order = torch.argsort(mon.util)
        weak_idx = order[:n_weak].tolist()
        strong_idx = order[-n_weak:].tolist()
        d = model.cfg.d
        x = xb.reshape(-1, d)
        loss, cnt = 0.0, 0
        for layer in model.layers:
            for wi, si in zip(weak_idx, strong_idx):
                if layer.experts[wi].special:
                    continue          # 特区探索型不参与蒸馏，保留探索性
                out_w = layer.experts[wi](x)
                out_s = layer.experts[si](x).detach()
                loss = loss + F.mse_loss(out_w, out_s)
                cnt += 1
        return (loss / max(1, cnt), n_weak)


# ================================================================ 主体模型
class MarvisMoE(nn.Module):
    """可缩放 MoE 框架：E 专家 x L 层 + 共享专家 + 治理循环（v6 修复版）。"""
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
        for layer in self.layers:
            x, aux, cap_l = layer(x, topk=self.council.k, cap=self.council.cap,
                                  lbda=self.council.lbda, adjust=adjust)
            aux_total = aux_total + aux + self.cfg.cap_w * cap_l
        x = x.mean(dim=1)
        return self.head(x), aux_total

    # ---------------- 训练循环（实践-认识-再实践） ----------------
    def fit(self, X, y, epochs=6, batch_size=256, lr=2e-3, gov=True,
            council_every=20, edu_every=50, seed=2026, val_data=None):
        set_seed(seed)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n = X.shape[0]
        n_batch = max(1, n // batch_size)
        history = dict(loss=[], acc=[], val_acc=[], std=[], aux=[], events=[])

        for ep in range(epochs):
            perm = torch.randperm(n)
            for bi in range(n_batch):
                idx = perm[bi * batch_size:(bi + 1) * batch_size]
                xb, yb = X[idx], y[idx]
                opt.zero_grad()
                logits, aux = self(xb)
                ce = F.cross_entropy(logits, yb)
                total = ce + self.council.lbda * aux

                # 教育部：软蒸馏 loss（loss 形式，不破坏 Adam 状态与专家多样性）
                if gov and self.mon.steps % edu_every == 0:
                    dloss, nw = self.education.soft_loss(self, self.mon, xb)
                    total = total + self.council.edu * dloss
                    dloss_v = dloss.item() if torch.is_tensor(dloss) else float(dloss)
                    history['events'].append(('educate', self.mon.steps,
                                              {'weak': nw, 'distill': dloss_v}))

                total.backward()
                opt.step()

                with torch.no_grad():
                    acc = (logits.argmax(-1) == yb).float().mean().item()
                    # Monitor 统计：所有层、使用实际路由决策（含容量修正与议会 k）
                    util = torch.zeros(self.cfg.E)
                    ent_sum, n_layers = 0.0, len(self.layers)
                    for layer in self.layers:
                        util.scatter_add_(0, layer.last_idx2.reshape(-1),
                                          torch.ones(layer.last_idx2.numel()))
                        p = layer.last_full_p
                        ent = -(p * p.clamp_min(1e-9).log()).sum(-1).mean()
                        ent_sum += float(ent)
                    entropy = ent_sum / max(1, n_layers)
                    self.mon.update(util, ce.item(), entropy)

                history['loss'].append(ce.item())
                history['acc'].append(acc)
                history['aux'].append(aux.item())

                # 治理循环：每 council_every 步议会投票（民主集中制）
                if gov and self.mon.steps % council_every == 0:
                    params = self.council.step(self.mon)
                    history['events'].append(('council', self.mon.steps, params.copy()))

                history['std'].append(self.mon.util_std)

            # 每 epoch 末验证集评估（泛化指标）
            if val_data is not None:
                Xv, yv = val_data
                self.eval()
                with torch.no_grad():
                    lv, _ = self(Xv)
                    va = (lv.argmax(-1) == yv).float().mean().item()
                self.train()
                history['val_acc'].append(va)

        history['final_util_std'] = self.mon.util_std
        history['final_active'] = self.mon.active_frac
        return history


# ================================================================ 同参数量密集 baseline
class DenseBaseline(nn.Module):
    """同参数量密集 MLP：用于公平验证 MoE 路由/稀疏开销与收益。"""
    def __init__(self, d, out_dim, n_params_target, n_layers=4):
        super().__init__()
        best = None
        for w in [32, 48, 64, 80, 96, 112, 128, 160, 192, 256]:
            params = d * w + (n_layers - 2) * w * w + w * out_dim
            if best is None or abs(params - n_params_target) < abs(best[1] - n_params_target):
                best = (w, params)
        w, params = best
        self.n_params = params
        layers = [nn.Linear(d, w), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(w, w), nn.SiLU()]
        layers.append(nn.Linear(w, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = x.mean(dim=1)
        return self.net(x)

    def fit(self, X, y, epochs=6, batch_size=256, lr=2e-3, seed=2026, val_data=None):
        set_seed(seed)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n = X.shape[0]
        n_batch = max(1, n // batch_size)
        hist = dict(loss=[], acc=[], val_acc=[])
        for ep in range(epochs):
            perm = torch.randperm(n)
            for bi in range(n_batch):
                idx = perm[bi * batch_size:(bi + 1) * batch_size]
                xb, yb = X[idx], y[idx]
                opt.zero_grad()
                logits = self(xb)
                ce = F.cross_entropy(logits, yb)
                ce.backward()
                opt.step()
                with torch.no_grad():
                    acc = (logits.argmax(-1) == yb).float().mean().item()
                hist['loss'].append(ce.item())
                hist['acc'].append(acc)
            if val_data is not None:
                Xv, yv = val_data
                self.eval()
                with torch.no_grad():
                    lv = self(Xv)
                    va = (lv.argmax(-1) == yv).float().mean().item()
                self.train()
                hist['val_acc'].append(va)
        return hist


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
        base = centers[c][None, :] + rng.standard_normal((per, d)) * 0.8
        mix = centers[(c + 1) % k][None, :] + rng.standard_normal((per, d)) * 0.8
        w = rng.uniform(0.2, 0.4, (per, 1))
        X.append(base * (1 - w) + mix * w)
        y.append(np.full(per, c))
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return torch.from_numpy(X), torch.from_numpy(y)


def _train_gov(cfg, Xtr, ytr, Xva, yva, gov, epochs=12, batch_size=256, lr=2e-3):
    set_seed(2026)
    model = MarvisMoE(cfg)
    h = model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size, lr=lr, gov=gov,
                  council_every=20, edu_every=50, val_data=(Xva, yva))
    return model, h


def _report(label, h, model=None, extra=""):
    last = min(50, len(h['acc']))
    print(f"  [{label}] 训练acc={np.mean(h['acc'][-last:]):.4f} "
          f"验证acc={h['val_acc'][-1]:.4f} " if h['val_acc'] else f"  [{label}] 训练acc={np.mean(h['acc'][-last:]):.4f} ")
    print(f"  末段损失={np.mean(h['loss'][-last:]):.4f} 负载std={h['final_util_std']:.4f} "
          f"活跃={h['final_active']:.2%} {extra}")


# ================================================================ 演示入口
def main():
    set_seed(2026)
    print("=" * 78)
    print("Marvis MoE v6（审查修复版）：E 路由专家 + S 共享专家 + 治理循环 + baseline")
    print("=" * 78)

    d, E, S, L, K = 16, 64, 2, 2, 8
    cfg = Config(d=d, h=32, E=E, S=S, L=L, topk=6, out_dim=K)
    X, y = make_synthetic(n=4096, d=d, k=K, seed=2026)
    # 类别混合拆分（make_synthetic 按类别顺序堆叠，直接切分会造成验证集类别缺失）
    perm_idx = torch.randperm(X.shape[0])
    X, y = X[perm_idx], y[perm_idx]
    n_tr = 3072
    Xtr, ytr, Xva, yva = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    n_params = sum(p.numel() for p in MarvisMoE(cfg).parameters())
    print(f"架构: E={E} 路由, S={S} 共享, L={L} 层, topk=6, 参数量={n_params:,} | 数据: train={n_tr} val={n_tr and X.shape[0]-n_tr}")

    results = {}
    for gov in (False, True):
        print(f"\n>>> 训练中：治理={'开' if gov else '关'}  (epochs=12, batch=256, lr=2e-3)")
        model, h = _train_gov(cfg, Xtr, ytr, Xva, yva, gov)
        last = min(50, len(h['acc']))
        print(f"  训练acc: {np.mean(h['acc'][-last:]):.4f} | 验证acc: {h['val_acc'][-1]:.4f}")
        print(f"  末段损失: {np.mean(h['loss'][-last:]):.4f} | 负载std: {h['final_util_std']:.4f} | 活跃: {h['final_active']:.2%}")
        if gov:
            print(f"  治理事件: {len(h['events'])} 次 (council={sum(1 for e in h['events'] if e[0]=='council')}, educate={sum(1 for e in h['events'] if e[0]=='educate')})")
            print(f"  末次治理参: lbda={model.council.lbda:.3f} cap={model.council.cap:.3f} edu={model.council.edu:.3f} k={model.council.k}")
        results[gov] = h

    # 同参数量密集 baseline
    print(f"\n>>> 训练中：同参数量密集 MLP baseline（约 {n_params:,} 参数）")
    set_seed(2026)
    dense = DenseBaseline(d, K, n_params)
    hd = dense.fit(Xtr, ytr, epochs=12, batch_size=256, lr=2e-3, val_data=(Xva, yva))
    last = min(50, len(hd['acc']))
    print(f"  训练acc: {np.mean(hd['acc'][-last:]):.4f} | 验证acc: {hd['val_acc'][-1]:.4f}")
    print(f"  末段损失: {np.mean(hd['loss'][-last:]):.4f} | 实际参数: {dense.n_params:,}")
    results['dense'] = hd

    print("\n" + "=" * 78)
    print("公平对比（同 seed=2026，同数据流，验证集泛化指标）")
    print("=" * 78)
    print(f"{'指标':<16}{'治理关':>14}{'治理开':>14}{'密集基线':>14}")
    rows = [
        ('验证acc', lambda h: h['val_acc'][-1]),
        ('训练acc', lambda h: np.mean(h['acc'][-50:])),
        ('末段损失', lambda h: np.mean(h['loss'][-50:])),
        ('负载std', lambda h: h['final_util_std']),
        ('活跃专家比', lambda h: h['final_active']),
    ]
    for name, f in rows:
        print(f"{name:<16}{f(results[False]):>14.4f}{f(results[True]):>14.4f}"
              f"{f(results['dense']) if name not in ('负载std', '活跃专家比') else '-':>14}")


def demo_medium():
    """中等规模扩展性验证：d=64, h=128, E=16, L=3（CPU 可负担，检验放大路径）。"""
    set_seed(2026)
    print("=" * 78)
    print("demo_medium：d=64 h=128 E=16 S=2 L=3 topk=4（扩展性验证）")
    print("=" * 78)
    d, E, S, L, K = 64, 16, 2, 3, 8
    cfg = Config(d=d, h=128, E=E, S=S, L=L, topk=4, out_dim=K)
    X, y = make_synthetic(n=4096, d=d, k=K, seed=2026)
    # 类别混合拆分（make_synthetic 按类别顺序堆叠，直接切分会造成验证集类别缺失）
    perm_idx = torch.randperm(X.shape[0])
    X, y = X[perm_idx], y[perm_idx]
    n_tr = 3072
    Xtr, ytr, Xva, yva = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
    n_params = sum(p.numel() for p in MarvisMoE(cfg).parameters())
    print(f"参数量={n_params:,}")
    results = {}
    for gov in (False, True):
        print(f"\n>>> 训练中：治理={'开' if gov else '关'}  (epochs=4, batch=256, lr=2e-3)")
        model, h = _train_gov(cfg, Xtr, ytr, Xva, yva, gov, epochs=4)
        last = min(50, len(h['acc']))
        print(f"  训练acc: {np.mean(h['acc'][-last:]):.4f} | 验证acc: {h['val_acc'][-1]:.4f} "
              f"| 负载std: {h['final_util_std']:.4f} | 活跃: {h['final_active']:.2%}")
        if gov:
            print(f"  末次治理参: lbda={model.council.lbda:.3f} cap={model.council.cap:.3f} "
                  f"edu={model.council.edu:.3f} k={model.council.k}")
        results[gov] = h
    print(f"\n验证acc对比：治理关={results[False]['val_acc'][-1]:.4f} "
          f"治理开={results[True]['val_acc'][-1]:.4f}")


if __name__ == '__main__':
    main()
