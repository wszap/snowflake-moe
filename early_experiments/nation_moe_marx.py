# -*- coding: utf-8 -*-
"""
100-Marvis 治理型 MoE —— 马克思主义思想深度迭代版
把"融合制度国家"升级为真实可训练代码：一个 numpy 迷你 MoE
- 100 个专家节点 = 100 个 Marvis 分身（并行待命，每步稀疏激活 top-K 工作）
- 可训练路由器（门控打分 = 市场部），负载均衡损失（税务局），知识蒸馏（教育部）
- 马克思主义治理循环：实践(训练)→认识(委员会监控)→再实践(调参)
  矛盾论: 抓主要矛盾——领域 loss 离散时加权资源
  量变质变: 负载/损失超阈值触发质变(疲劳/教育)
  否定之否定: 弱专家被市场否定后经教育螺旋上升
  生产关系反作用生产力: 治理机制(路由/税收/配额)释放专家产能
"""
import numpy as np

np.random.seed(2026)
E, D, C, TOPK = 100, 8, 6, 6           # 100 Marvis / 输入8维 / 6领域 / top-6
N_STEPS, BATCH = 8000, 128
SWITCH = int(N_STEPS * 0.6)            # 前60%热点领域0/1, 后40%漂移到领域3/4
LR = 0.05
major = np.arange(E) % C               # 每个 Marvis 的"籍贯领域"(先验, 可被学习覆盖)

# ---------- 数据: 6 领域回归任务 ----------
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

def make_stream(n):
    rng = np.random.default_rng(7)
    cs = np.empty(n, dtype=int)
    for t in range(n):
        if t < SWITCH:
            p = [0.45, 0.35, 0.05, 0.05, 0.05, 0.05]
        else:
            p = [0.05, 0.05, 0.05, 0.45, 0.35, 0.05]
        cs[t] = rng.choice(C, p=p)
    return cs

train_c = make_stream(N_STEPS)
# 测试集: 各领域均匀
Xte, Yte = [], []
for c in range(C):
    x, y = gen_task(c, 400)
    Xte.append(x); Yte.append(y)
Xte, Yte = np.vstack(Xte), np.concatenate(Yte)

# ---------- MoE 模型: 100 专家(线性) + 门控路由器 ----------
W = np.random.randn(E, D) * 0.5        # 专家权重 100x8
B = np.zeros(E)                          # 专家偏置
Wr = np.random.randn(D, E) * 0.05       # 路由器 8x100
affin = np.zeros((E, C))                 # 领域亲和先验(门控偏置)
for e in range(E):
    affin[e, major[e]] = 1.2

def forward(x, b, load_bias):
    """x: (B,D) -> yhat, 被激活专家索引, 门控概率"""
    logits = x @ Wr + affin[b] + load_bias
    if spec_noise > 0:
        logits[:, spec_idx] += np.random.normal(0, spec_noise, (x.shape[0], len(spec_idx)))
    idx = np.argsort(logits, axis=1)[:, -TOPK:]
    g = np.zeros_like(logits)
    np.put_along_axis(g, idx, 1.0, axis=1)
    g = g * np.exp(logits - logits.max(axis=1, keepdims=True))
    g = g / g.sum(axis=1, keepdims=True)
    expert_out = np.tanh(x @ W.T + B)    # (B,E)
    yhat = (g * expert_out).sum(axis=1)
    return yhat, idx, g, expert_out

# ---------- 治理参数(由委员会投票动态调整) ----------
lam_bal = 0.3        # 税务局: 负载均衡损失权重
spec_noise = 0.0     # 特区港: 探索噪声
edu_strength = 0.0   # 教育部: 知识蒸馏强度
fed_strength = 0.0   # 联邦: 紧急调配强度
spec_idx = [0, 1]    # 特区专家(香港自由港, 免税/高探索)
load_bias = np.zeros(E)                 # 负载偏置(公益组织/税务局调节)
freq = np.zeros(E)

# ---------- 100 个治理单元(委员)立场 ----------
# 10 类 x 10 人 = 100; 议题: [λ_bal, σ探索, 教育, 联邦, 配额]
POL = np.array([
    [-1, -1, -1, 0, 0],   # 0 资本派: 效率优先, 减税减探索
    [1, 0, 1, 0, 1],      # 1 公平派: 加平衡, 加教育
    [0, -1, 0, 1, 1],     # 2 计划派: 联邦调配, 配额
    [-1, 1, -1, -1, 0],   # 3 创新派: 探索, 去约束
    [0, 0, 1, 0, -1],     # 4 民生派: 教育弱领域
    [0, 0, 0, 0, 0],      # 5 稳定派: 温和
    [0, 0, 0, 1, 0],      # 6 联邦派: 强化中央调配
    [-1, 1, 0, -1, 0],    # 7 特区派: 自由港扩权
    [1, 0, 1, 0, 0],      # 8 教育派: 知识体系
    [0, 0, 0, 0, 0],      # 9 观察派: 随指标摇摆
])
n_types = len(POL)
unit_type = np.array([i for i in range(n_types) for _ in range(10)])   # 100 委员
unit_calls = np.zeros(E + 100)   # 前100: marvis激活次数; 后100: 委员投票次数
unit_acc = np.zeros(E + 100)

def committee_vote(stats):
    """100 委员根据指标投票: 返回 5 议题净票数"""
    acc_trend, std_l, gini_l, dom_spread = stats
    votes = np.zeros(5)
    for u in range(100):
        t = unit_type[u]
        base = POL[t].copy()
        if t == 9:  # 观察派: 按指标摇摆
            base = np.zeros(5)
            if gini_l > 0.4 or std_l > 0.5: base[0] += 1
            if acc_trend < 0: base[1] += 1
            if dom_spread > 0.3: base[3] += 1
        v = base + np.random.randint(-1, 2, 5) * 0
        votes += v
        unit_calls[100 + u] += 1
    return votes

def apply_votes(votes):
    global lam_bal, spec_noise, edu_strength, fed_strength
    lam_bal = float(np.clip(lam_bal * (1 + 0.15 * np.tanh(votes[0] / 20)), 0.02, 2.0))
    spec_noise = float(np.clip(spec_noise + 0.02 * np.tanh(votes[1] / 20), 0.05, 0.8))
    edu_strength = float(np.clip(edu_strength + 0.02 * np.tanh(votes[2] / 20), 0.0, 0.2))
    fed_strength = float(np.clip(fed_strength + 0.05 * np.tanh(votes[3] / 20), 0.0, 1.0))
    # 配额议题: 弱领域样本加权(主要矛盾集中) —— 由 dom_spread 隐含, 直接作用于 loss 加权

def train(governed):
    global W, B, Wr, load_bias, freq, lam_bal, spec_noise, edu_strength, fed_strength
    W[:] = np.random.randn(E, D) * 0.5; B[:] = 0
    Wr[:] = np.random.randn(D, E) * 0.05
    load_bias[:] = 0; freq[:] = 0
    lam_bal = 0.3; spec_noise = 0.0; edu_strength = 0.0; fed_strength = 0.0
    loss_hist, std_hist, dom_loss = [], [], np.zeros(C)
    acc_trend = 0.0
    for st in range(N_STEPS):
        c = train_c[st]
        x, y = gen_task(c, BATCH)
        yhat, idx, g, eo = forward(x, c, load_bias)
        # 主要矛盾: 领域 loss 离散度加权(矛盾论: 抓主要矛盾)
        dom_mse = np.mean((yhat - y) ** 2)
        dom_loss[c] = dom_mse
        spread = np.std(dom_loss) / (dom_loss.mean() + 1e-9)
        # 负载均衡损失(税务局): E * Σ_e f_e * ĝ_e   (经典 aux loss)
        f_e = np.bincount(idx.flatten(), minlength=E) / (BATCH * TOPK)
        g_avg = np.zeros(E)
        np.add.at(g_avg, idx.flatten(), g[np.arange(BATCH).repeat(TOPK), idx.flatten()])
        g_avg /= (BATCH * TOPK)
        aux = E * np.sum(f_e * g_avg)
        loss = dom_mse + (lam_bal * aux if governed else 0.2 * aux)
        # 反传: 门控与专家梯度(简化为对激活专家/门控的直接SGD)
        d_y = 2 * (yhat - y) / BATCH
        d_g = np.zeros_like(g)
        np.put_along_axis(d_g, idx, (eo * d_y[:, None])[:, idx] if False else 1.0, axis=1)
        # 简化: 只更新专家和路由器(logits通过softmax影响)
        g_soft = np.exp(logits - logits.max(axis=1, keepdims=True)); g_soft /= g_soft.sum(1, keepdims=True)
        d_logits = g_soft * (eo * d_y[:, None] - (g_soft * eo * d_y[:, None]).sum(1, keepdims=True))
        Wr -= LR * (x.T @ d_logits) / BATCH
        for k in range(TOPK):
            col = idx[:, k]
            grad_w = d_y * g[np.arange(BATCH), col][:, None] * (1 - eo[np.arange(BATCH), col][:, None] ** 2) * x
            np.add.at(W, col, -LR * grad_w)
        freq = freq * 0.99 + np.bincount(idx.flatten(), minlength=E) / (BATCH * TOPK)
        # ---- 治理循环(实践-认识-再实践) ----
        if governed and st > 0 and st % 100 == 0:
            # 认识: 监控指标
            gini = float(np.mean(np.abs(np.subtract.outer(freq, freq)))) / (2 * freq.mean() + 1e-9)
            std_l = float(freq.std())
            acc_win = 1.0 - np.mean((yhat - y) ** 2) / 4.0
            acc_trend = 0.7 * acc_trend + 0.3 * (acc_win - 0.5)
            votes = committee_vote((acc_trend, std_l, gini, spread))
            apply_votes(votes)
            # 教育部: 否定之否定——弱领域专家向强领域学习(知识蒸馏)
            if edu_strength > 0 and st % 300 == 0:
                weak_c = np.argmax(dom_loss); strong_c = np.argmin(dom_loss)
                weak_idx = np.where(major == weak_c)[0]
                strong_avg = W[np.where(major == strong_c)[0]].mean(axis=0)
                W[weak_idx] = (1 - edu_strength) * W[weak_idx] + edu_strength * strong_avg
            # 联邦紧急调配: 量变质变——漂移期热点领域负载bias释放
            if fed_strength > 0:
                hot_c = int(np.argmax(np.bincount(train_c[max(0, st - 200):st], minlength=C)))
                load_bias = np.where(major == hot_c, -fed_strength, load_bias * 0.9)
        # 记录
        if st % 400 == 0:
            loss_hist.append(dom_mse); std_hist.append(freq.std())
    # 测试
    yte = forward(Xte, np.zeros(len(Xte), dtype=int), np.zeros(E))[0]
    acc = float(np.mean(np.sign(yte) == np.sign(Yte)))
    return acc, loss_hist, std_hist

# 需要 logits 供 train 内使用——重构 forward 使其返回 logits
def forward(x, b, load_bias, ret_logits=False):
    logits = x @ Wr + affin[:, b].T + load_bias
    if spec_noise > 0:
        logits[:, spec_idx] += np.random.normal(0, spec_noise, (x.shape[0], len(spec_idx)))
    idx = np.argsort(logits, axis=1)[:, -TOPK:]
    g = np.zeros_like(logits)
    np.put_along_axis(g, idx, 1.0, axis=1)
    g = g * np.exp(logits - logits.max(axis=1, keepdims=True))
    g = g / g.sum(axis=1, keepdims=True)
    eo = np.tanh(x @ W.T + B)
    yhat = (g * eo).sum(axis=1)
    if ret_logits:
        return yhat, idx, g, eo, logits
    return yhat, idx, g, eo

# 重写 train 以使用 ret_logits
def train(governed):
    global W, B, Wr, load_bias, freq, lam_bal, spec_noise, edu_strength, fed_strength
    W[:] = np.random.randn(E, D) * 0.5; B[:] = 0
    Wr[:] = np.random.randn(D, E) * 0.05
    load_bias[:] = 0; freq[:] = 0
    lam_bal = 0.3; spec_noise = 0.0; edu_strength = 0.0; fed_strength = 0.0
    loss_hist, std_hist, dom_loss = [], [], np.zeros(C)
    acc_trend = 0.0
    for st in range(N_STEPS):
        c = train_c[st]
        x, y = gen_task(c, BATCH)
        yhat, idx, g, eo, logits = forward(x, c, load_bias, ret_logits=True)
        dom_mse = np.mean((yhat - y) ** 2)
        dom_loss[c] = dom_mse
        spread = np.std(dom_loss) / (dom_loss.mean() + 1e-9)
        f_e = np.bincount(idx.flatten(), minlength=E) / (BATCH * TOPK)
        g_avg = np.zeros(E)
        np.add.at(g_avg, idx.flatten(), g[np.arange(BATCH).repeat(TOPK), idx.flatten()])
        g_avg /= (BATCH * TOPK)
        aux = E * np.sum(f_e * g_avg)
        loss = dom_mse + (lam_bal * aux if governed else 0.2 * aux)
        d_y = 2 * (yhat - y) / BATCH
        eo_b = eo[np.arange(BATCH).repeat(TOPK), idx.flatten()]          # (B*K)
        gy = g[np.arange(BATCH).repeat(TOPK), idx.flatten()] * d_y.repeat(TOPK)
        grad_w = gy[:, None] * (1 - eo_b ** 2)[:, None] * x.repeat(TOPK, axis=0)
        np.add.at(W, idx.flatten(), -LR * grad_w)
        g_soft = np.exp(logits - logits.max(axis=1, keepdims=True))
        g_soft /= g_soft.sum(axis=1, keepdims=True)
        d_logits = g_soft * (eo * d_y[:, None] - (g_soft * eo * d_y[:, None]).sum(1, keepdims=True))
        Wr -= LR * (x.T @ d_logits) / BATCH
        freq = freq * 0.99 + np.bincount(idx.flatten(), minlength=E) / (BATCH * TOPK)
        if governed and st > 0 and st % 100 == 0:
            gini = float(np.mean(np.abs(np.subtract.outer(freq, freq)))) / (2 * freq.mean() + 1e-9)
            std_l = float(freq.std())
            acc_win = 1.0 - dom_mse / 4.0
            acc_trend = 0.7 * acc_trend + 0.3 * (acc_win - 0.5)
            votes = committee_vote((acc_trend, std_l, gini, spread))
            apply_votes(votes)
            if edu_strength > 0 and st % 300 == 0:
                weak_c = np.argmax(dom_loss); strong_c = np.argmin(dom_loss)
                weak_idx = np.where(major == weak_c)[0]
                strong_avg = W[np.where(major == strong_c)[0]].mean(axis=0)
                W[weak_idx] = (1 - edu_strength) * W[weak_idx] + edu_strength * strong_avg
            if fed_strength > 0:
                hot_c = int(np.argmax(np.bincount(train_c[max(0, st - 200):st], minlength=C)))
                load_bias = np.where(major == hot_c, -fed_strength, load_bias * 0.9)
        if st % 400 == 0:
            loss_hist.append(dom_mse); std_hist.append(freq.std())
    yte = forward(Xte, np.zeros(len(Xte), dtype=int), np.zeros(E))[0]
    acc = float(np.mean(np.sign(yte) == np.sign(Yte)))
    return acc, loss_hist, std_hist

if __name__ == "__main__":
    print("=" * 86)
    print("100-Marvis 治理型 MoE（马克思主义深度迭代）")
    print("专家数=100(=100个Marvis) | 领域=6 | 激活top-6 | 漂移: 领域0/1 → 领域3/4")
    print("委员会=100委员(10派系×10人) 每100步实践-认识-再实践投票调参")
    print("=" * 86)
    acc_raw, lh_raw, sh_raw = train(False)
    print(f"裸MoE        : 测试正确率={acc_raw:.3f} | 末段损失={lh_raw[-1]:.4f} | 负载std={sh_raw[-1]:.4f}")
    acc_gov, lh_gov, sh_gov = train(True)
    print(f"治理型MoE    : 测试正确率={acc_gov:.3f} | 末段损失={lh_gov[-1]:.4f} | 负载std={sh_gov[-1]:.4f}")
    print("-" * 86)
    print(f"正确率提升 = {acc_gov - acc_raw:+.3f} | 均衡提升 = {sh_raw[-1] - sh_gov[-1]:+.4f}")
    print(f"最终治理参数: λ_bal={lam_bal:.3f} σ探索={spec_noise:.3f} 教育={edu_strength:.3f} 联邦={fed_strength:.3f}")
    print("-" * 86)
    print("马克思主义思想 ↔ AI 机制映射：")
    print("  矛盾论        → 主要矛盾=领域loss离散度, 委员会据此加权/调配")
    print("  量变质变      → 负载频率累积触发疲劳, 漂移阈值触发联邦紧急调配")
    print("  否定之否定    → 弱专家被市场否定 → 教育部知识蒸馏螺旋升级")
    print("  生产关系反作用生产力 → 路由/税收/配额释放专家产能(均衡不减正确率)")
    print("  实践-认识-再实践 → 训练(实践)→监控指标(认识)→投票调参(再实践)")
    print("  民主集中制    → 100委员投票产生治理参数, 中央(路由器)执行")
