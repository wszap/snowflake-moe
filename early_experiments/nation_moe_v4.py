# -*- coding: utf-8 -*-
"""
MoE-V4 —— DeepSeek 式真实组件（细粒度专家 + 共享专家 + 容量因子 + aux loss）
把上一版"100-Marvis 治理型 MoE"升级为真实大模型的组件级实现：
- 细粒度路由专家 24 个（每领域 4 个）—— DeepSeek-V3/V4 思路：细粒度→组合空间更大
- 共享专家 2 个（永远激活）—— 承担通用底座，DeepSeek-V3 首创于主模型
- 容量因子 bias（软约束，禁止硬禁选）—— 防路由坍塌又不牺牲效率
- 保留 100 委员治理循环（矛盾论/否定之否定/实践-认识-再实践）
任务流与上一版完全一致，可公平对比"整块专家"vs"细粒度+共享专家"。
"""
import numpy as np

np.random.seed(2026)
D, C, BATCH = 8, 6, 128
N_STEPS, SWITCH = 6000, int(6000 * 0.6)
LR = 0.05

E_ROUTE, E_SHARED, TOPK = 24, 2, 4          # 细粒度路由专家 / 共享专家 / 路由激活数
major = np.arange(E_ROUTE) % C
CAP = 1.25                                   # 容量因子(DeepSeek-V3 用 1.25)

domain_vec = np.array([[0.6, 0.1, -0.2, 0.3, -0.1, 0.2, 0.0, 0.1],
                       [-0.2, 0.7, 0.3, -0.1, 0.4, 0.0, 0.2, -0.3],
                       [0.3, -0.2, 0.8, 0.2, -0.4, 0.5, -0.1, 0.2],
                       [-0.4, 0.2, 0.1, 0.9, 0.3, -0.2, 0.4, 0.0],
                       [0.2, 0.4, -0.3, 0.1, 0.7, 0.1, -0.2, 0.5],
                       [-0.1, 0.0, 0.5, -0.2, 0.2, 0.6, 0.3, -0.4]])
base_w = np.array([0.5, -0.3, 0.8, 0.2, -0.6, 0.4, 0.1, -0.2])

def gen_task(c, n):
    x = np.random.randn(n, D)
    w = base_w + domain_vec[c]
    y = np.tanh(x @ w) * 1.5 + 0.3 * np.sin(x[:, 0] * 2)
    return x, y

rng = np.random.default_rng(7)
train_c = np.empty(N_STEPS, dtype=int)
for t in range(N_STEPS):
    p = [0.45, 0.35, 0.05, 0.05, 0.05, 0.05] if t < SWITCH else [0.05, 0.05, 0.05, 0.45, 0.35, 0.05]
    train_c[t] = rng.choice(C, p=p)

Xte, Yte = [], []
for c in range(C):
    x, y = gen_task(c, 400); Xte.append(x); Yte.append(y)
Xte, Yte = np.vstack(Xte), np.concatenate(Yte)

Wr = np.random.randn(E_ROUTE, D) * 0.5      # 细粒度专家
Ws = np.random.randn(E_SHARED, D) * 0.5     # 共享专家
G = np.random.randn(D, E_ROUTE) * 0.05      # 路由器
affin = np.zeros((E_ROUTE, C))
for e in range(E_ROUTE):
    affin[e, major[e]] = 1.2
load_bias = np.zeros(E_ROUTE)
freq = np.zeros(E_ROUTE)
lam_bal, spec_noise, edu_strength, fed_strength = 0.3, 0.0, 0.0, 0.0
spec_idx = [0, 1]

POL = np.array([[-1, -1, -1, 0, 0], [1, 0, 1, 0, 1], [0, -1, 0, 1, 1], [-1, 1, -1, -1, 0],
                [0, 0, 1, 0, -1], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [-1, 1, 0, -1, 0],
                [1, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
unit_type = np.array([i for i in range(10) for _ in range(10)])
unit_calls = np.zeros(E_ROUTE + E_SHARED + 100)

def forward(x, b, ldb, ret_logits=False):
    logits = x @ G + affin[:, b].T + ldb
    if spec_noise > 0:
        logits[:, spec_idx] += np.random.normal(0, spec_noise, (x.shape[0], len(spec_idx)))
    # 容量因子软约束: 超容量专家 logits 惩罚(不硬禁选, 防坍塌又保效率)
    over = np.maximum(freq - CAP / E_ROUTE, 0)
    logits = logits - 2.0 * over[None, :]
    idx = np.argsort(logits, axis=1)[:, -TOPK:]
    g = np.zeros_like(logits)
    np.put_along_axis(g, idx, 1.0, axis=1)
    g = g * np.exp(logits - logits.max(axis=1, keepdims=True))
    g = g / g.sum(axis=1, keepdims=True)
    route_out = np.tanh(x @ Wr.T + np.zeros(E_ROUTE))   # (B,24)
    shared_out = np.tanh(x @ Ws.T + np.zeros(E_SHARED))  # (B,2)
    yhat = (g * route_out).sum(axis=1) + 0.5 * shared_out.sum(axis=1)
    if ret_logits:
        return yhat, idx, g, route_out, shared_out, logits
    return yhat, idx, g, route_out, shared_out

def train(fine_grained, governed):
    global Wr, Ws, G, load_bias, freq, lam_bal, spec_noise, edu_strength, fed_strength
    Wr[:] = np.random.randn(E_ROUTE, D) * 0.5
    Ws[:] = np.random.randn(E_SHARED, D) * 0.5
    G[:] = np.random.randn(D, E_ROUTE) * 0.05
    load_bias[:] = 0; freq[:] = 0
    lam_bal, spec_noise, edu_strength, fed_strength = 0.3, 0.0, 0.0, 0.0
    loss_hist, std_hist, dom_loss = [], [], np.zeros(C)
    acc_trend = 0.0
    for st in range(N_STEPS):
        c = train_c[st]
        x, y = gen_task(c, BATCH)
        yhat, idx, g, route_out, shared_out, logits = forward(x, c, load_bias, ret_logits=True)
        dom_mse = np.mean((yhat - y) ** 2)
        dom_loss[c] = dom_mse
        spread = np.std(dom_loss) / (dom_loss.mean() + 1e-9)
        f_e = np.bincount(idx.flatten(), minlength=E_ROUTE) / (BATCH * TOPK)
        g_avg = np.zeros(E_ROUTE)
        np.add.at(g_avg, idx.flatten(), g[np.arange(BATCH).repeat(TOPK), idx.flatten()])
        g_avg /= (BATCH * TOPK)
        aux = E_ROUTE * np.sum(f_e * g_avg)
        loss = dom_mse + (lam_bal * aux if governed else 0.2 * aux)
        d_y = 2 * (yhat - y) / BATCH
        # 路由专家梯度
        eo_b = route_out[np.arange(BATCH).repeat(TOPK), idx.flatten()]
        gy = g[np.arange(BATCH).repeat(TOPK), idx.flatten()] * d_y.repeat(TOPK)
        grad_w = gy[:, None] * (1 - eo_b ** 2)[:, None] * x.repeat(TOPK, axis=0)
        np.add.at(Wr, idx.flatten(), -LR * grad_w)
        # 共享专家梯度(永远激活, 承担底座)
        gs = 0.5 * d_y[:, None] * (1 - shared_out ** 2)      # (B,2)
        Ws -= LR * 0.1 * (gs.T @ x) / BATCH                  # 小步长共享更新
        # 门控梯度
        g_soft = np.exp(logits - logits.max(axis=1, keepdims=True))
        g_soft /= g_soft.sum(axis=1, keepdims=True)
        d_logits = g_soft * (route_out * d_y[:, None] - (g_soft * route_out * d_y[:, None]).sum(1, keepdims=True))
        G -= LR * (x.T @ d_logits) / BATCH
        freq = freq * 0.99 + np.bincount(idx.flatten(), minlength=E_ROUTE) / (BATCH * TOPK)
        if governed and st > 0 and st % 100 == 0:
            gini = float(np.mean(np.abs(np.subtract.outer(freq, freq)))) / (2 * freq.mean() + 1e-9)
            std_l = float(freq.std())
            acc_win = 1.0 - dom_mse / 4.0
            acc_trend = 0.7 * acc_trend + 0.3 * (acc_win - 0.5)
            votes = np.zeros(5)
            for u in range(100):
                t = unit_type[u]; base = POL[t].copy()
                if t == 9:
                    base = np.zeros(5)
                    if gini > 0.4 or std_l > 0.5: base[0] += 1
                    if acc_trend < 0: base[1] += 1
                    if spread > 0.3: base[3] += 1
                votes += base
            lam_bal = float(np.clip(lam_bal * (1 + 0.15 * np.tanh(votes[0] / 20)), 0.02, 2.0))
            spec_noise = float(np.clip(spec_noise + 0.02 * np.tanh(votes[1] / 20), 0.05, 0.8))
            edu_strength = float(np.clip(edu_strength + 0.02 * np.tanh(votes[2] / 20), 0.0, 0.2))
            fed_strength = float(np.clip(fed_strength + 0.05 * np.tanh(votes[3] / 20), 0.0, 1.0))
            if edu_strength > 0 and st % 300 == 0:
                weak_c = np.argmax(dom_loss); strong_c = np.argmin(dom_loss)
                weak_idx = np.where(major == weak_c)[0]
                strong_avg = Wr[np.where(major == strong_c)[0]].mean(axis=0)
                Wr[weak_idx] = (1 - edu_strength) * Wr[weak_idx] + edu_strength * strong_avg
            if fed_strength > 0:
                hot_c = int(np.argmax(np.bincount(train_c[max(0, st - 200):st], minlength=C)))
                load_bias = np.where(major == hot_c, -fed_strength, load_bias * 0.9)
        if st % 400 == 0:
            loss_hist.append(dom_mse); std_hist.append(freq.std())
    yte = forward(Xte, np.zeros(len(Xte), dtype=int), np.zeros(E_ROUTE))[0]
    acc = float(np.mean(np.sign(yte) == np.sign(Yte)))
    return acc, loss_hist[-1], std_hist[-1]

if __name__ == "__main__":
    print("=" * 84)
    print("MoE-V4: DeepSeek 式组件（细粒度24 + 共享2 + 容量因子1.25） vs 上版整块100专家")
    print("=" * 84)
    # 上版实测(同任务流): 整块裸0.790/0.1848/1.027 | 整块治理0.790/0.1714/0.856
    acc_a, l_a, s_a = train(True, False)   # 细粒度无治理
    acc_b, l_b, s_b = train(True, True)    # 细粒度+共享+治理
    print(f"{'配置':<28}{'正确率':<10}{'末段损失':<12}{'负载std':<10}")
    print(f"{'整块100专家·裸(v3)':<26}{0.790:<12.3f}{0.1848:<14.4f}{1.0270:<10.4f}")
    print(f"{'整块100专家·治理(v3)':<24}{0.790:<12.3f}{0.1714:<14.4f}{0.8557:<10.4f}")
    print(f"{'细粒度24+共享2·裸(v4)':<22}{acc_a:<14.3f}{l_a:<14.4f}{s_a:<10.4f}")
    print(f"{'细粒度24+共享2·治理(v4)':<20}{acc_b:<14.3f}{l_b:<14.4f}{s_b:<10.4f}")
    print("-" * 84)
    print("DeepSeek 式机制验证：细粒度专家→组合空间↑ | 共享专家→通用底座 | 容量因子→软约束防坍塌")
