# -*- coding: utf-8 -*-
"""
把大模型建造成一个"制度混合国家"：市场 + 计划 + 联邦 + 议会 + 特区 + 公益
- 资本主义成分 : 专家按"能力分 + 累积资本"竞价（能者多得，马太效应）
- 社会主义成分 : 累进税再分配（对头部资本征税补贴弱势领域）+ 公共专家保底服务
- 中国模式     : 计划委员会——热点领域"举国体制"式资源倾斜（联邦紧急调配）
- 美国模式     : 联邦制——平时领域自治，热点突变时联邦宣布"紧急状态"集中调配
- 英国模式     : 议会制——领域代表每周期投票审议税率与补贴方案
- 香港模式     : 2 个"自由港特区"专家——免税收、不受计划约束，纯市场自由竞争
- 公益组织     : NGO 监测各领域"人民满意度/资源失衡"，发布报告影响议会决策（软性协调）

人民 = 知识体系（任务请求流），评价 = 人民满意度（正确率）。
与独裁制/民主集中制/企业制在同一世界对比。
"""
import numpy as np

np.random.seed(2026)
E, C, K = 24, 6, 4
N = 8000
SWITCH = int(N * 0.6)
CAP_WIN = 100
CAP = 17
NOISE = 0.02
PUB_N = 2                      # 公共专家（保底服务）
SPECIAL = [0, 1]               # 特区自由港专家（不交税、不受计划约束）

# ---- 专家真实能力 ----
q = 0.90 - 0.006 * np.arange(E)
# 特区专家是"通用型人才"：基础更强、无领域加成（自由探索）
q[SPECIAL] = [0.965, 0.955]
major = np.arange(E) % C
aff = np.zeros((E, C))
for i in range(E):
    if i not in SPECIAL:
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

# ---------- 制度混合制 ----------
def run_mixed():
    capital = np.full(E, 100.0)          # 市场资本
    use = np.zeros(E)
    acc = np.zeros(N)
    load = np.zeros(E)
    fatigue = np.ones(E)
    pub_q = np.full(PUB_N, 0.82)         # 公共专家质量
    tax = 0.12                           # 累进税率（议会审议后调整）
    social_fund = 0.0                    # 加班重税累积的社保基金
    emergency = False
    prev_dist = None
    # 周期统计
    per_domain_ok = np.zeros(C)
    per_domain_cnt = np.zeros(C)
    for t in range(N):
        # 每窗结算疲劳
        if t % CAP_WIN == 0 and t > 0:
            fatigue = np.where(load > CAP, 0.7, 1.0)
            load[:] = 0
        # 每 500 token 一个"财政周期"：NGO 报告 → 议会审议 → 累进税再分配
        if t > 0 and t % 500 == 0:
            # ① 公益组织发布失衡报告（满意度 + 资源占比）
            sat = np.where(per_domain_cnt > 0, per_domain_ok / np.maximum(per_domain_cnt, 1), 0.5)
            res_share = np.bincount(major, weights=load, minlength=C) / max(load.sum(), 1)
            gini = float(np.mean(np.abs(np.subtract.outer(capital, capital)))) / (2 * capital.mean())
            imbalance = float(np.std(sat) + np.std(res_share))
            # ② 议会审议：失衡大 → 通过加税提案（社会主义再分配力度加大）
            tax = min(0.12 + 0.20 * imbalance + 0.15 * (gini > 0.25), 0.35)
            # ③ 累进税 + 补贴弱势：头部征税，按领域满意度缺口返补
            paid = capital * tax
            capital -= paid
            weak = np.argsort(sat)[:2]                       # 满意度最低的两个领域
            for dd in weak:
                idx = [i for i in range(E) if major[i] == dd]
                bonus = (paid.sum() + social_fund) / 2 / max(len(idx), 1)   # 社保基金一起返补
                for i in idx:
                    capital[i] += bonus
            social_fund = 0.0
            # ⑤ 反垄断法：资本超额部分强制上缴（特区免税但同样受竞争法约束）
            for i in range(E):
                cap_lim = 300 if i in SPECIAL else 400
                if capital[i] > cap_lim:
                    social_fund += (capital[i] - cap_lim) * 0.5
                    capital[i] = cap_lim + (capital[i] - cap_lim) * 0.5
            # ④ 联邦状态：检测热点突变（与上窗需求分布比）
            dist = np.bincount(tasks[max(0, t - 500):t], minlength=C) / 500
            if prev_dist is not None:
                emergency = float(np.abs(dist - prev_dist).sum()) > 0.5
            prev_dist = dist
            per_domain_ok[:] = 0
            per_domain_cnt[:] = 0
        c = tasks[t]
        # ---- 市场竞价（资本主义：能力 + 资本加成，特区专家参与）----
        domain = [i for i in range(E) if major[i] == c]
        cand = domain + SPECIAL                                  # 本领域 + 特区
        s = np.array([true_skill(i, c) for i in cand]) * fatigue[cand]
        s += np.random.normal(0, NOISE, len(cand))
        s += 0.0005 * np.array([capital[i] for i in cand])       # 资本加成（马太）
        s *= (1.0 - (np.array([load[i] for i in cand]) / CAP).clip(0, 1)) ** 2  # 工时约束(劳动法,平方级)
        # 市场席：按竞价取前 2（特区自由竞争可挤入）；过载则"劳务派遣"跨领域补员
        order = np.argsort(s)[::-1]
        picked = []
        for idx in order:
            if len(picked) == 2:
                break
            i = cand[idx]
            if load[i] < CAP * 1.5:
                picked.append(i)
            else:
                backup = [j for j in np.argsort(load) if j not in picked and j not in cand and load[j] < CAP]
                picked.append(backup[0] if backup else i)
        # ---- 计划保底（社会主义：公共专家基本服务，弱势领域兜底）----
        picked.append(-1 - int(t % PUB_N))                       # 公共专家
        # ---- 联邦/公益协调席（美国联邦主义 + NGO 公益协调）----
        if emergency:
            # 联邦紧急调配：全国负载最低的强专家（免疲劳、集中资源）
            backup = [i for i in np.argsort(load) if i not in picked and i >= 0]
            pick2 = backup[0] if backup else SPECIAL[0]
        else:
            # 公益协调：优先选本领域负载最低者（区域平衡），没有就跨领域借调
            dom_sorted = sorted(domain, key=lambda i: load[i])
            pick2 = next((i for i in dom_sorted if i not in picked), None)
            if pick2 is None:
                backup = [i for i in np.argsort(load) if i not in picked and i >= 0]
                pick2 = backup[0] if backup else SPECIAL[0]
        picked.append(pick2)
        # ---- 执行与结算 ----
        for i in picked:
            if i >= 0:
                load[i] += 1
                use[i] += 1
                ok = answer_ok(i, c)
                per_domain_ok[c] += ok
                per_domain_cnt[c] += 1
                # 市场结算：答对赚资本，答错亏资本；超时工作交"加班重税"进社保基金
                if ok:
                    capital[i] += 5.0
                    acc[t] += 1
                else:
                    capital[i] -= 3.0
                if load[i] > CAP and i not in SPECIAL:           # 劳动法：超时重税(特区免税)
                    fine = 4.0
                    capital[i] -= fine
                    social_fund += fine
                if i in SPECIAL:
                    pass                                          # 香港模式：免税
                else:
                    capital[i] = max(capital[i] - 0.3, 10)        # 普通专家维持成本
            else:
                ok = np.random.rand() < pub_q[-1 - i]             # 公共专家
                acc[t] += ok
        acc[t] /= K
    return acc, use, capital

# ---------- 复用：独裁制 / 民主集中制 / 企业制（从 nation_vs_enterprise 移植）----------
def run_autocrat():
    logits = np.zeros(E)
    use = np.zeros(E)
    fatigue = np.ones(E)
    acc = np.zeros(N)
    load = np.zeros(E)
    for t in range(N):
        if t % CAP_WIN == 0 and t > 0:
            fatigue = np.where(load > CAP, 0.7, 1.0)
            load[:] = 0
        c = tasks[t]
        s = np.array([true_skill(i, c) for i in range(E)]) * fatigue
        s += np.random.normal(0, NOISE, E) + logits
        top = np.argsort(s)[-K:]
        for i in top:
            load[i] += 1
            use[i] += 1
            ok = answer_ok(i, c)
            if ok:
                logits[i] = min(logits[i] + 0.05, 0.5)
                acc[t] += 1
            else:
                logits[i] = max(logits[i] - 0.025, -0.5)
        acc[t] /= K
    return acc, use, logits

def run_democracy():
    qq = q.copy()
    use = np.zeros(E)
    acc = np.zeros(N)
    load = np.zeros(E)
    perf = np.zeros(E)
    fatigue = np.ones(E)
    for t in range(N):
        if t % CAP_WIN == 0 and t > 0:
            fatigue = np.where(load > CAP, 0.7, 1.0)
            load[:] = 0
        if t > 0 and t % 2000 == 0:
            order = np.argsort(perf)
            bad = order[:E // 5]
            best = order[-1]
            for i in bad:
                qq[i] = np.mean(qq[major == major[i]]) * 0.95
            qq[best] = min(qq[best] + 0.004, 0.99)
            perf[:] = 0
        c = tasks[t]
        domain = np.where(major == c)[0]
        votes = np.zeros(E)
        for rep in domain:
            s_rep = np.array([true_skill(i, c) for i in domain]) * fatigue[domain]
            s_rep += np.random.normal(0, NOISE, len(domain))
            votes[domain[np.argmax(s_rep)]] += 1
        d_rank = np.argsort(votes[domain])[::-1]
        elected = list(domain[d_rank[:3]])
        s_all = np.array([true_skill(i, c) for i in range(E)]) * fatigue
        s_all += np.random.normal(0, NOISE, E)
        s_all *= (1.0 - 0.4 * (load / CAP).clip(0, 1))
        cand = np.argsort(s_all)[::-1]
        for i in cand:
            if i not in elected and len(elected) < K:
                elected.append(i)
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

def run_enterprise():
    qq = q.copy()
    shared_q = np.full(PUB_N, 0.82)
    use = np.zeros(E)
    acc = np.zeros(N)
    load = np.zeros(E)
    budget = np.full(C, K)
    domain_q = np.zeros(C)
    domain_cnt = np.zeros(C)
    bad_streak = np.zeros(C)
    for t in range(N):
        if t > 0 and t % 500 == 0:
            hist = tasks[max(0, t - 500):t]
            p_c = np.bincount(hist, minlength=C) / len(hist)
            total = K * 500
            new_b = np.floor(p_c * total).astype(int)
            with np.errstate(divide="ignore", invalid="ignore"):
                dq = np.where(domain_cnt > 0, domain_q / np.maximum(domain_cnt, 1), 0.5)
            order = np.argsort(dq)
            for dd in order[:2]:
                new_b[dd] = max(int(new_b[dd] * 0.7), K)
                bad_streak[dd] += 1
            for dd in order[-2:]:
                new_b[dd] += int(K * 500 * 0.06)
                bad_streak[dd] = 0
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
        s_d = np.array([true_skill(i, c) for i in domain]) * (load[domain] < CAP)
        s_d += np.random.normal(0, NOISE, len(domain))
        picked = []
        for idx in np.argsort(s_d)[::-1]:
            if budget[c] > 0:
                picked.append(domain[idx])
                budget[c] -= 1
            if len(picked) == K:
                break
        while len(picked) < K:
            picked.append(-1 - np.random.randint(PUB_N))
        for i in picked:
            if i >= 0:
                load[i] += 1
                use[i] += 1
                ok = answer_ok(i, c)
                domain_q[c] += ok
                domain_cnt[c] += 1
            else:
                ok = np.random.rand() < shared_q[-1 - i]
            acc[t] += ok
        acc[t] /= K
    return acc, use, qq

def stats(name, acc, use):
    s1 = acc[:SWITCH].mean()
    s2 = acc[SWITCH:].mean()
    tot = acc.mean()
    std = use.std()
    active = (use > 0).sum()
    rec = None
    base = s1 * 0.95
    for t in range(SWITCH, N - 100, 10):
        if acc[t:t + 100].mean() >= base:
            rec = t - SWITCH
            break
    print(f"{name:10s} 阶段1={s1:.3f} 阶段2={s2:.3f} 总体={tot:.3f} | 负载std={std:7.1f} 活跃={active:2d}/24 | 恢复={rec if rec is not None else '未达'}")
    return s1, s2, tot, std, active, rec

print("=" * 84)
print("世界: 24专家/6领域, 含2个免税特区专家; 前60%热点领域0, 后40%漂移到领域3")
print("混合制机构: 市场竞价(资本主义) + 累进税再分配(社会主义) + 公共保底(计划) +")
print("            联邦紧急调配(美国) + 议会审议税率(英国) + 免税特区(香港) + NGO报告(公益)")
print("=" * 84)
for name, fn in [("独裁制", run_autocrat), ("民主集中", run_democracy), ("企业制", run_enterprise), ("混合制", run_mixed)]:
    acc, use, _ = fn()
    stats(name, acc, use)
print("-" * 84)
print("恢复=漂移后正确率回到阶段1的95%所需token数; 混合制应综合各方优点")
