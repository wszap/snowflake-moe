# -*- coding: utf-8 -*-
"""
把大模型看做一个国家：三种"治理体制"的路由实验
- 独裁制    : 单一门控说了算（基线）
- 民主集中制: 领域代表投票 + 中央统筹否决 + 换届任期（模拟中国政权模式的组织逻辑）
- 企业制    : 专家=产线，知识=订单，预算+需求预测+KPI考核+资源再分配（企业资源管理）

世界设定:
- 24 个专家（=官员/产线），6 个领域（=部门/产品线），每领域 4 个专家
- 专家 i 质量 q_i 线性递减；对主领域有加成（专长分工）
- 任务流: 前 60% 时间热点在领域0（70%），后 40% 突然漂移到领域3（70%）——考验体制韧性
- 每 token 激活 K=4 个专家；专家超负荷会疲劳（质量下降）——资源稀缺性
- 门控/代表/管理层看到的分数都带噪声（信息不对称）

评测: 正确率、负载均衡(标准差)、活跃专家数、漂移后的恢复速度
"""
import numpy as np

np.random.seed(2026)
E, C, K = 24, 6, 4
N = 8000
SWITCH = int(N * 0.6)
CAP_WIN = 100          # 负荷统计窗口
CAP = 17               # 每窗最大服务量（24专家*100token*4激活/24≈16.7）
NOISE = 0.02           # 感知噪声
SHARED_N = 2           # 企业制的共享服务中心专家数

# ---- 专家真实能力 ----
q = 0.90 - 0.006 * np.arange(E)
major = np.arange(E) % C
aff = np.zeros((E, C))
for i in range(E):
    aff[i, major[i]] = 0.12

def true_skill(i, c):
    return q[i] + aff[i, c]

def task_stream():
    rng = np.random.default_rng(7)
    tasks = np.empty(N, dtype=int)
    for t in range(N):
        if t < SWITCH:
            p = [0.70, 0.20, 0.03, 0.02, 0.03, 0.02]
        else:
            p = [0.02, 0.03, 0.03, 0.70, 0.20, 0.02]
        tasks[t] = rng.choice(C, p=p)
    return tasks

tasks = task_stream()

def answer_ok(i, c):
    return np.random.rand() < true_skill(i, c)

# ---------- 体制 1：独裁制 ----------
def run_autocrat():
    logits = np.zeros(E)                 # 独裁者的"学习权重"
    use = np.zeros(E)
    fatigue = np.ones(E)
    acc = np.zeros(N)
    load = np.zeros(E)
    res = []
    for t in range(N):
        if t % CAP_WIN == 0 and t > 0:
            fatigue = np.where(load > CAP, 0.7, 1.0)   # 上窗超载者疲劳
            load[:] = 0
        c = tasks[t]
        s = np.array([true_skill(i, c) for i in range(E)]) * fatigue
        s += np.random.normal(0, NOISE, E) + logits     # 感知+经验
        top = np.argsort(s)[-K:]
        for i in top:
            load[i] += 1
            use[i] += 1
            ok = answer_ok(i, c)
            if ok:
                logits[i] = min(logits[i] + 0.05, 0.5)   # 答对加分（偏爱强者，有界）
                acc[t] += 1
            else:
                logits[i] = max(logits[i] - 0.025, -0.5)
        acc[t] /= K
        res.append((fatigue.copy(), logits.copy()))
    return acc, use, res

# ---------- 体制 2：民主集中制（中国政权模式的组织逻辑）----------
def run_democracy():
    qq = q.copy()                        # 现任专家质量（换届会变）
    use = np.zeros(E)
    acc = np.zeros(N)
    load = np.zeros(E)
    perf = np.zeros(E)                   # 本届绩效
    fatigue = np.ones(E)
    for t in range(N):
        if t % CAP_WIN == 0 and t > 0:
            fatigue = np.where(load > CAP, 0.7, 1.0)
            load[:] = 0
        # 换届：每 2000 token 一届
        if t > 0 and t % 2000 == 0:
            order = np.argsort(perf)
            bad = order[:E // 5]         # 绩效垫底 20% 下台
            best = order[-1]
            for i in bad:
                qq[i] = np.mean(qq[major == major[i]]) * 0.95   # 新官上任，略弱于同领域均值
            qq[best] = min(qq[best] + 0.004, 0.99)              # 连任者小幅晋升
            perf[:] = 0
        c = tasks[t]
        domain = np.where(major == c)[0]                       # 本领域 4 名代表
        # ① 代表投票：每名代表对 4 名候选人独立打分（各有独立噪声）
        votes = np.zeros(E)
        for rep in domain:
            s_rep = np.array([true_skill(i, c) for i in domain]) * fatigue[domain]
            s_rep += np.random.normal(0, NOISE, len(domain))
            votes[domain[np.argmax(s_rep)]] += 1               # 每代表投一票
        # ② 领域席位：得票前 3 当选
        d_rank = np.argsort(votes[domain])[::-1]
        elected = list(domain[d_rank[:3]])
        # ③ 中央统筹：另 1 席从全国按"能力×负载折扣"调配
        s_all = np.array([true_skill(i, c) for i in range(E)]) * fatigue
        s_all += np.random.normal(0, NOISE, E)
        s_all *= (1.0 - 0.4 * (load / CAP).clip(0, 1))         # 负载折扣：中央向清闲专家倾斜
        cand = np.argsort(s_all)[::-1]
        for i in cand:
            if i not in elected and len(elected) < K:
                elected.append(i)
        # ④ 中央否决：当选者若超载，否决并换人（负载最低的合格专家）
        for idx in range(K):
            if load[elected[idx]] >= CAP:
                backup = np.argsort(load)
                for j in backup:
                    if j not in elected:
                        elected[idx] = j
                        break
        for i in elected:
            load[i] += 1
            use[i] += 1
            ok = answer_ok(i, c)
            perf[i] += 1.0 if ok else 0.0
            acc[t] += ok
        acc[t] /= K
    return acc, use, qq

# ---------- 体制 3：企业制（知识资源当企业资源管理）----------
def run_enterprise():
    qq = q.copy()
    # 2 个共享服务中心专家（通用、无领域加成、质量中上）
    shared_q = np.full(SHARED_N, 0.82)
    use = np.zeros(E)
    acc = np.zeros(N)
    load = np.zeros(E)
    budget = np.full(C, K)                # 每个领域池的周期预算（初始均分）
    domain_q = np.zeros(C)                # 领域 KPI（累计正确率）
    domain_cnt = np.zeros(C)
    bad_streak = np.zeros(C)
    for t in range(N):
        # 每 500 token 一个经营周期：需求预测 → 预算再分配 → KPI 考核
        if t > 0 and t % 500 == 0:
            hist = tasks[max(0, t - 500):t]
            p_c = np.bincount(hist, minlength=C) / len(hist)    # 需求预测
            total = K * 500
            new_b = np.floor(p_c * total).astype(int)           # 以销定产
            # 考核：按领域平均正确率排名，末 2 位削减预算转给前 2 位
            with np.errstate(divide="ignore", invalid="ignore"):
                dq = np.where(domain_cnt > 0, domain_q / np.maximum(domain_cnt, 1), 0.5)
            order = np.argsort(dq)
            for dd in order[:2]:
                new_b[dd] = max(int(new_b[dd] * 0.7), K)        # 整改：砍 30%
                bad_streak[dd] += 1
            for dd in order[-2:]:
                new_b[dd] += int(K * 500 * 0.06)                # 头部加码
                bad_streak[dd] = 0
            # 连续两期垫底：停产重组（专家质量重置）
            for dd in range(C):
                if bad_streak[dd] >= 2:
                    idx = np.where(major == dd)[0]
                    for i in idx:
                        qq[i] = 0.90
                    bad_streak[dd] = 0
            budget = new_b
            domain_q[:] = 0
            domain_cnt[:] = 0
            load[:] = 0
        c = tasks[t]
        domain = np.where(major == c)[0]
        # 本领域池内按感知分选（预算内）
        s_d = np.array([true_skill(i, c) for i in domain]) * (load[domain] < CAP)
        s_d += np.random.normal(0, NOISE, len(domain))
        picked = []
        for idx in np.argsort(s_d)[::-1]:
            if budget[c] > 0:
                picked.append(domain[idx])
                budget[c] -= 1
            if len(picked) == K:
                break
        # 预算不足（超卖）：转共享服务中心
        while len(picked) < K:
            picked.append(-1 - np.random.randint(SHARED_N))     # 负索引代表共享专家
        for i in picked:
            if i >= 0:
                load[i] += 1
                use[i] += 1
                ok = answer_ok(i, c)
                domain_q[c] += ok
                domain_cnt[c] += 1
            else:
                ok = np.random.rand() < shared_q[-1 - i]        # 共享专家
            acc[t] += ok
        acc[t] /= K
    return acc, use, qq

def stats(name, acc, use):
    s1 = acc[:SWITCH].mean()
    s2 = acc[SWITCH:].mean()
    tot = acc.mean()
    std = use.std()
    active = (use > 0).sum()
    # 恢复速度：阶段2滑动100窗口达到 s1*0.95 的首个位置
    rec = None
    base = s1 * 0.95
    for t in range(SWITCH, N - 100, 10):
        if acc[t:t + 100].mean() >= base:
            rec = t - SWITCH
            break
    print(f"{name:8s} 阶段1={s1:.3f} 阶段2={s2:.3f} 总体={tot:.3f} | 负载std={std:.1f} 活跃={active}/24 | 恢复={rec if rec is not None else '未达'}")
    return s1, s2, tot, std, active, rec

print("=" * 78)
print("世界设定: 24专家/6领域, 前60%热点在领域0, 后40%漂移到领域3; 每窗超载17次则疲劳")
print("=" * 78)
for name, fn in [("独裁制", run_autocrat), ("民主集中", run_democracy), ("企业制", run_enterprise)]:
    acc, use, _ = fn()
    stats(name, acc, use)
print("-" * 78)
print("说明: 恢复=漂移后正确率回到阶段1的95%所需token数; 越小韧性越强")

# 运行结果（seed 2026 / 任务流 seed 7）:
#   独裁制   阶段1=0.946 阶段2=0.873 总体=0.917 | 负载std=2440.5 活跃=7/24  | 恢复=40
#   民主集中 阶段1=0.880 阶段2=0.883 总体=0.881 | 负载std=27.6   活跃=24/24 | 恢复=0
#   企业制   阶段1=0.935 阶段2=0.912 总体=0.926 | 负载std=1006.8 活跃=24/24 | 恢复=170
# 对照 seed（2027/8）趋势一致：独裁漂移掉点最多且坍缩；民主最稳最均衡但质量略低；
# 企业总正确率最高，但产能调整滞后，恢复最慢。
