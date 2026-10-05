# -*- coding: utf-8 -*-
"""
融合制度国家 · 8 个 Marvis 分身协同实验
把一个大模型国家拆给 8 个分身治理，融合资本主义+社会主义+中国+美国+香港+英国制度：
  1 市场部Marvis    —— 资本主义：按能力+资本竞价路由(能者多得)
  2 计划委Marvis    —— 社会主义/中国：公共专家保底、计划配额
  3 联邦政府Marvis  —— 美国：平时领域自治，热点突变宣布紧急状态集中调配
  4 议会Marvis      —— 英国：领域代表审议税率与补贴方案
  5 特区Marvis      —— 香港：2个自由港专家，免税、自由竞争、不受计划约束
  6 公益组织Marvis  —— NGO：区域(领域)失衡监测，软性协调倡议
  7 教育部Marvis    —— 共同富裕：特区创新知识向弱势领域溢出(培训)
  8 总统Marvis      —— 汇总各分身提案，最终拍板路由

人民=知识体系(任务流)，评价=人民满意度(正确率)。
与独裁制/民主集中制/企业制在同一世界对比。
"""
import numpy as np

np.random.seed(2026)
E, C, K = 24, 6, 4
N = 8000
SWITCH = int(N * 0.6)
CAP_WIN, CAP, NOISE, PUB_N = 100, 17, 0.02, 2
SPECIAL = [0, 1]

q_base = 0.90 - 0.006 * np.arange(E)
q_base[SPECIAL] = [0.965, 0.955]          # 特区专家：通用型人才
major = np.arange(E) % C
aff = np.zeros((E, C))
for i in range(E):
    if i not in SPECIAL:
        aff[i, major[i]] = 0.12

def skill(qq, i, c):
    return qq[i] + aff[i, c]

def tasks():
    rng = np.random.default_rng(7)
    out = np.empty(N, dtype=int)
    for t in range(N):
        p = [0.70, 0.20, 0.03, 0.02, 0.03, 0.02] if t < SWITCH else [0.02, 0.03, 0.03, 0.70, 0.20, 0.02]
        out[t] = rng.choice(C, p=p)
    return out
TASKS = tasks()

# ---------- 8 个分身：每个独立类，记录调用/采纳次数 ----------
class Marvis:
    def __init__(self, name, role):
        self.name, self.role = name, role
        self.calls = self.accepted = 0

class MarketMarvis(Marvis):                       # 1 市场部
    def __init__(self): super().__init__("市场部Marvis", "竞价路由(资本主义)")
    def bid(self, w, c, domain, spec):
        self.calls += 1
        cand = domain + spec
        s = np.array([skill(w.qq, i, c) for i in cand]) * w.fatigue[cand]
        s += np.random.normal(0, NOISE, len(cand))
        s += 0.0005 * np.array([w.capital[i] for i in cand])
        s *= (1.0 - (np.array([w.load[i] for i in cand]) / CAP).clip(0, 1)) ** 2
        order = np.argsort(s)[::-1]
        out = []
        for idx in order:
            if len(out) == 2: break
            i = cand[idx]
            if w.load[i] < CAP * 1.5:
                out.append(i)
            else:
                bk = [j for j in np.argsort(w.load) if j not in out and j not in cand and w.load[j] < CAP]
                out.append(bk[0] if bk else i)
        self.accepted += len(out)
        return out

class PlannerMarvis(Marvis):                      # 2 计划委
    def __init__(self): super().__init__("计划委Marvis", "公共保底(社会主义)")
    def ensure(self, w, t):
        self.calls += 1; self.accepted += 1
        return [-1 - int(t % PUB_N)]

class NGOmarvis(Marvis):                          # 6 公益组织
    def __init__(self): super().__init__("公益组织Marvis", "区域调度协调(NGO)")
    def coordinate(self, w, domain, taken):
        self.calls += 1
        dom_sorted = sorted(domain, key=lambda i: w.load[i])
        pick = next((i for i in dom_sorted if i not in taken), None)
        if pick is None:
            bk = [i for i in np.argsort(w.load) if i not in taken and i >= 0]
            pick = bk[0] if bk else SPECIAL[0]
        self.accepted += 1
        return [pick]
    def report(self, w):
        sat = np.where(w.dom_cnt > 0, w.dom_ok / np.maximum(w.dom_cnt, 1), 0.5)
        res = np.bincount(major, weights=w.load, minlength=C) / max(w.load.sum(), 1)
        gini = float(np.mean(np.abs(np.subtract.outer(w.capital, w.capital)))) / (2 * w.capital.mean())
        return sat, res, float(np.std(sat) + np.std(res)), gini

class FederationMarvis(Marvis):                   # 3 联邦政府
    def __init__(self): super().__init__("联邦政府Marvis", "联邦紧急调配(美国)")
    def detect(self, w, t):
        self.calls += 1
        dist = np.bincount(TASKS[max(0, t - 500):t], minlength=C) / 500
        if w.prev_dist is not None:
            w.emergency = float(np.abs(dist - w.prev_dist).sum()) > 0.5
        w.prev_dist = dist
    def mobilize(self, w, taken):
        self.calls += 1
        bk = [i for i in np.argsort(w.load) if i not in taken and i >= 0]
        pick = bk[0] if bk else SPECIAL[0]
        self.accepted += 1
        return [pick]

class ParliamentMarvis(Marvis):                   # 4 议会
    def __init__(self): super().__init__("议会Marvis", "审议税率(英国)")
    def review(self, w, imb, gini):
        self.calls += 1
        w.tax = min(0.12 + 0.20 * imb + 0.15 * (gini > 0.25), 0.35)
        self.accepted += 1

class TaxMarvis(Marvis):                          # 税务并入总统? 单独设立更清晰 -> 作为总统的执行职能
    def __init__(self): super().__init__("税务局Marvis", "累进税+社保+反垄断")
    def levy(self, w, weak):
        self.calls += 1
        paid = w.capital * w.tax
        w.capital -= paid
        for dd in weak:
            idx = [i for i in range(E) if major[i] == dd]
            bonus = (paid.sum() + w.social) / 2 / max(len(idx), 1)
            for i in idx: w.capital[i] += bonus
        w.social = 0.0
        for i in range(E):
            lim = 300 if i in SPECIAL else 400
            if w.capital[i] > lim:
                w.social += (w.capital[i] - lim) * 0.5
                w.capital[i] = lim + (w.capital[i] - lim) * 0.5
        self.accepted += 1

class EducatorMarvis(Marvis):                     # 7 教育部
    def __init__(self): super().__init__("教育部Marvis", "知识溢出(共同富裕)")
    def teach(self, w, weak):
        self.calls += 1
        spec_avg = w.qq[SPECIAL].mean()
        for dd in weak:
            for i in range(E):
                if major[i] == dd and i not in SPECIAL:
                    w.qq[i] += (spec_avg - w.qq[i]) * 0.15
        self.accepted += 1

class PresidentMarvis(Marvis):                    # 8 总统
    def __init__(self): super().__init__("总统Marvis", "汇总拍板(中央)")
    def decide(self, w, proposals):
        self.calls += 1
        self.accepted += 4
        return proposals

def run_fusion():
    np.random.seed(2026)
    w = type("W", (), {"qq": q_base.copy(), "capital": np.full(E, 100.0), "use": np.zeros(E),
                       "load": np.zeros(E), "fatigue": np.ones(E), "tax": 0.12, "social": 0.0,
                       "emergency": False, "prev_dist": None, "pub_q": np.full(PUB_N, 0.82),
                       "dom_ok": np.zeros(C), "dom_cnt": np.zeros(C)})()
    marvises = [MarketMarvis(), PlannerMarvis(), FederationMarvis(), ParliamentMarvis(),
                TaxMarvis(), NGOmarvis(), EducatorMarvis(), PresidentMarvis()]
    acc = np.zeros(N)
    for t in range(N):
        if t % CAP_WIN == 0 and t > 0:
            w.fatigue = np.where(w.load > CAP, 0.7, 1.0); w.load[:] = 0
        if t > 0 and t % 500 == 0:
            ngo = next(m for m in marvises if isinstance(m, NGOmarvis))
            sat, res, imb, gini = ngo.report(w)
            parl = next(m for m in marvises if isinstance(m, ParliamentMarvis)); parl.review(w, imb, gini)
            taxm = next(m for m in marvises if isinstance(m, TaxMarvis))
            weak = np.argsort(sat)[:2]; taxm.levy(w, weak)
            edu = next(m for m in marvises if isinstance(m, EducatorMarvis)); edu.teach(w, weak)
            fed = next(m for m in marvises if isinstance(m, FederationMarvis)); fed.detect(w, t)
            w.dom_ok[:] = 0; w.dom_cnt[:] = 0
        c = TASKS[t]
        domain = [i for i in range(E) if major[i] == c]
        mkt = next(m for m in marvises if isinstance(m, MarketMarvis)); picked = mkt.bid(w, c, domain, SPECIAL)
        pln = next(m for m in marvises if isinstance(m, PlannerMarvis)); picked += pln.ensure(w, t)
        ngo = next(m for m in marvises if isinstance(m, NGOmarvis))
        fed = next(m for m in marvises if isinstance(m, FederationMarvis))
        if w.emergency:
            picked += fed.mobilize(w, picked)
        else:
            picked += ngo.coordinate(w, domain, picked)
        pres = next(m for m in marvises if isinstance(m, PresidentMarvis))
        picked = pres.decide(w, picked)
        for i in picked:
            if i >= 0:
                w.load[i] += 1; w.use[i] += 1
                ok = np.random.rand() < skill(w.qq, i, c)
                w.dom_ok[c] += ok; w.dom_cnt[c] += 1
                if ok:
                    w.capital[i] += 5.0; acc[t] += 1
                else:
                    w.capital[i] -= 3.0
                if w.load[i] > CAP and i not in SPECIAL:
                    w.capital[i] -= 4.0; w.social += 4.0
                if i not in SPECIAL:
                    w.capital[i] = max(w.capital[i] - 0.3, 10)
            else:
                acc[t] += np.random.rand() < w.pub_q[-1 - i]
        acc[t] /= K
    return acc, w.use, marvises

# ================= v2 迭代采优：10 Marvis 分身 =================
# 录取来源：独裁(能力logits) + 民主(强制轮换) + 企业(需求预测预算) + 融合(特区/NGO/教育/联邦/累进税)
class SARMarvis(Marvis):                          # 9 特区港（新增）
    def __init__(self): super().__init__("特区港Marvis", "自由港专家管理(香港)")
    def manage(self, w, c):
        self.calls += 1
        self.accepted += len(SPECIAL)
        return SPECIAL                                # 特区专家参与自由竞价（免税不受计划约束）

class MeteoMarvis(Marvis):                        # 10 气象局（新增，录取企业制）
    def __init__(self): super().__init__("气象局Marvis", "需求预测预算(企业制录取)")
    def forecast(self, w, t):
        self.calls += 1
        p_c = np.bincount(TASKS[max(0, t - 500):t], minlength=C) / 500
        w.budget = np.maximum((p_c * (K - 1) * 500).astype(int), K * 2)   # 领域市场配额(冷门保底)
        self.accepted += 1

def run_fusion_v2():
    np.random.seed(2026)
    w = type("W", (), {"qq": q_base.copy(), "capital": np.full(E, 100.0), "use": np.zeros(E),
                       "load": np.zeros(E), "fatigue": np.ones(E), "tax": 0.12, "social": 0.0,
                       "emergency": False, "prev_dist": None, "pub_q": np.full(PUB_N, 0.82),
                       "dom_ok": np.zeros(C), "dom_cnt": np.zeros(C),
                       "logits": np.zeros(E), "budget": np.full(C, K * 2)})()
    marvises = [MarketMarvis(), PlannerMarvis(), FederationMarvis(), ParliamentMarvis(),
                TaxMarvis(), NGOmarvis(), EducatorMarvis(), PresidentMarvis(),
                SARMarvis(), MeteoMarvis()]
    acc = np.zeros(N)
    for t in range(N):
        if t % CAP_WIN == 0 and t > 0:
            w.fatigue = np.where(w.load > CAP, 0.7, 1.0); w.load[:] = 0
        if t > 0 and t % 500 == 0:
            ngo = next(m for m in marvises if isinstance(m, NGOmarvis))
            sat, res, imb, gini = ngo.report(w)
            parl = next(m for m in marvises if isinstance(m, ParliamentMarvis)); parl.review(w, imb, gini)
            taxm = next(m for m in marvises if isinstance(m, TaxMarvis))
            weak = np.argsort(sat)[:2]; taxm.levy(w, weak)
            edu = next(m for m in marvises if isinstance(m, EducatorMarvis)); edu.teach(w, weak)
            fed = next(m for m in marvises if isinstance(m, FederationMarvis)); fed.detect(w, t)
            met = next(m for m in marvises if isinstance(m, MeteoMarvis)); met.forecast(w, t)   # 气象局
            w.dom_ok[:] = 0; w.dom_cnt[:] = 0
        c = TASKS[t]
        domain = [i for i in range(E) if major[i] == c]
        # ---- 市场竞价（v2：录取独裁logits + 民主强制轮换 + 企业配额）----
        sar = next(m for m in marvises if isinstance(m, SARMarvis))
        cand = domain + sar.manage(w, c)
        s = np.array([skill(w.qq, i, c) for i in cand]) * w.fatigue[cand]
        s += np.random.normal(0, NOISE, len(cand))
        s += 0.0005 * np.array([w.capital[i] for i in cand])
        s += np.array([w.logits[i] for i in cand])                              # 独裁录取:经验加成
        s *= (1.0 - (np.array([w.load[i] for i in cand]) / CAP).clip(0, 1)) ** 2
        order = np.argsort(s)[::-1]
        mkt = next(m for m in marvises if isinstance(m, MarketMarvis)); mkt.calls += 1
        picked = []
        for idx in order:
            if len(picked) == 2: break
            i = cand[idx]
            if w.load[i] < CAP * 1.3:                                           # 民主录取:强制轮换线
                picked.append(i)
            else:
                bk = [j for j in np.argsort(w.load) if j not in picked and j not in cand and w.load[j] < CAP]
                picked.append(bk[0] if bk else i)
        if w.budget[c] <= 0:                                                    # 企业录取:配额耗尽→跨区驰援
            picked = [j for j in np.argsort(w.load)[:2] if j not in picked]
        else:
            w.budget[c] -= 2
        mkt.accepted += len(picked)
        # ---- 计划保底 / 联邦-NGO协调 / 总统拍板（同v1）----
        pln = next(m for m in marvises if isinstance(m, PlannerMarvis)); picked += pln.ensure(w, t)
        ngo = next(m for m in marvises if isinstance(m, NGOmarvis))
        fed = next(m for m in marvises if isinstance(m, FederationMarvis))
        if w.emergency:
            picked += fed.mobilize(w, picked)
        else:
            picked += ngo.coordinate(w, domain, picked)
        pres = next(m for m in marvises if isinstance(m, PresidentMarvis)); picked = pres.decide(w, picked)
        for i in picked:
            if i >= 0:
                w.load[i] += 1; w.use[i] += 1
                ok = np.random.rand() < skill(w.qq, i, c)
                w.dom_ok[c] += ok; w.dom_cnt[c] += 1
                if ok:
                    w.capital[i] += 5.0; acc[t] += 1
                    w.logits[i] = min(w.logits[i] + 0.05, 0.5)                  # 独裁录取:能力积累
                else:
                    w.capital[i] -= 3.0
                    w.logits[i] = max(w.logits[i] - 0.025, -0.5)
                if w.load[i] > CAP and i not in SPECIAL:
                    w.capital[i] -= 4.0; w.social += 4.0
                if i not in SPECIAL:
                    w.capital[i] = max(w.capital[i] - 0.3, 10)
            else:
                acc[t] += np.random.rand() < w.pub_q[-1 - i]
        acc[t] /= K
    return acc, w.use, marvises

# ---------- 对照：独裁 / 民主 / 企业（同一世界移植） ----------
def run_autocrat():
    np.random.seed(2026)
    logits = np.zeros(E); use = np.zeros(E); fatigue = np.ones(E); acc = np.zeros(N); load = np.zeros(E)
    for t in range(N):
        if t % CAP_WIN == 0 and t > 0:
            fatigue = np.where(load > CAP, 0.7, 1.0); load[:] = 0
        c = TASKS[t]
        s = np.array([skill(q_base, i, c) for i in range(E)]) * fatigue
        s += np.random.normal(0, NOISE, E) + logits
        top = np.argsort(s)[-K:]
        for i in top:
            load[i] += 1; use[i] += 1
            ok = np.random.rand() < skill(q_base, i, c)
            if ok:
                logits[i] = min(logits[i] + 0.05, 0.5); acc[t] += 1
            else:
                logits[i] = max(logits[i] - 0.025, -0.5)
        acc[t] /= K
    return acc, use

def run_democracy():
    np.random.seed(2026)
    qq = q_base.copy(); use = np.zeros(E); acc = np.zeros(N); load = np.zeros(E)
    perf = np.zeros(E); fatigue = np.ones(E)
    for t in range(N):
        if t % CAP_WIN == 0 and t > 0:
            fatigue = np.where(load > CAP, 0.7, 1.0); load[:] = 0
        if t > 0 and t % 2000 == 0:
            order = np.argsort(perf); bad = order[:E // 5]; best = order[-1]
            for i in bad: qq[i] = np.mean(qq[major == major[i]]) * 0.95
            qq[best] = min(qq[best] + 0.004, 0.99); perf[:] = 0
        c = TASKS[t]
        domain = np.where(major == c)[0]
        votes = np.zeros(E)
        for rep in domain:
            s_rep = np.array([skill(qq, i, c) for i in domain]) * fatigue[domain]
            s_rep += np.random.normal(0, NOISE, len(domain))
            votes[domain[np.argmax(s_rep)]] += 1
        d_rank = np.argsort(votes[domain])[::-1]
        elected = list(domain[d_rank[:3]])
        s_all = np.array([skill(qq, i, c) for i in range(E)]) * fatigue
        s_all += np.random.normal(0, NOISE, E)
        s_all *= (1.0 - 0.4 * (load / CAP).clip(0, 1))
        cand = np.argsort(s_all)[::-1]
        for i in cand:
            if i not in elected and len(elected) < K: elected.append(i)
        for idx in range(K):
            if load[elected[idx]] >= CAP:
                for j in np.argsort(load):
                    if j not in elected: elected[idx] = j; break
        for i in elected:
            load[i] += 1; use[i] += 1
            ok = np.random.rand() < skill(qq, i, c); perf[i] += 1.0 if ok else 0.0; acc[t] += ok
        acc[t] /= K
    return acc, use

def run_enterprise():
    np.random.seed(2026)
    qq = q_base.copy(); use = np.zeros(E); acc = np.zeros(N); load = np.zeros(E)
    budget = np.full(C, K); dom_q = np.zeros(C); dom_cnt = np.zeros(C); bad = np.zeros(C)
    pub_q = np.full(PUB_N, 0.82)
    for t in range(N):
        if t > 0 and t % 500 == 0:
            hist = TASKS[max(0, t - 500):t]
            p_c = np.bincount(hist, minlength=C) / len(hist)
            new_b = np.floor(p_c * K * 500).astype(int)
            dq = np.where(dom_cnt > 0, dom_q / np.maximum(dom_cnt, 1), 0.5)
            order = np.argsort(dq)
            for dd in order[:2]:
                new_b[dd] = max(int(new_b[dd] * 0.7), K); bad[dd] += 1
            for dd in order[-2:]:
                new_b[dd] += int(K * 500 * 0.06); bad[dd] = 0
            for dd in range(C):
                if bad[dd] >= 2:
                    for i in np.where(major == dd)[0]: qq[i] = 0.90
                    bad[dd] = 0
            budget = new_b; dom_q[:] = 0; dom_cnt[:] = 0; load[:] = 0
        c = TASKS[t]
        domain = np.where(major == c)[0]
        s_d = np.array([skill(qq, i, c) for i in domain]) * (load[domain] < CAP)
        s_d += np.random.normal(0, NOISE, len(domain))
        picked = []
        for idx in np.argsort(s_d)[::-1]:
            if budget[c] > 0 and len(picked) < K:
                picked.append(domain[idx]); budget[c] -= 1
        while len(picked) < K:
            picked.append(-1 - np.random.randint(PUB_N))
        for i in picked:
            if i >= 0:
                load[i] += 1; use[i] += 1
                ok = np.random.rand() < skill(qq, i, c); dom_q[c] += ok; dom_cnt[c] += 1
            else:
                ok = np.random.rand() < pub_q[-1 - i]
            acc[t] += ok
        acc[t] /= K
    return acc, use

def stats(name, acc, use):
    s1, s2, tot = acc[:SWITCH].mean(), acc[SWITCH:].mean(), acc.mean()
    std, active = use.std(), (use > 0).sum()
    rec = None
    base = s1 * 0.95
    for t in range(SWITCH, N - 100, 10):
        if acc[t:t + 100].mean() >= base:
            rec = t - SWITCH; break
    print(f"{name:8s} 阶段1={s1:.3f} 阶段2={s2:.3f} 总体={tot:.3f} | 负载std={std:7.1f} 活跃={active:2d}/24 | 恢复={rec if rec is not None else '未达'}")
    return s1, s2, tot, std, active, rec

print("=" * 88)
print("融合制度国家 · 10 Marvis 分身协同（v2 迭代采优版）")
print("人民=知识体系(任务流) | 市场部(资本竞价) 计划委(公共保底) 联邦(紧急调配) 议会(税率)")
print("税务局(累进税+反垄断) 公益组织(区域协调) 教育部(知识溢出) 总统(汇总拍板)")
print("特区港(自由港专家管理) 气象局(需求预测预算) | v2录取：独裁经验+民主轮换+企业配额")
print("=" * 88)
marvises = None
for name, fn in [("独裁制", run_autocrat), ("民主集中", run_democracy), ("企业制", run_enterprise),
                 ("融合制", run_fusion), ("融合制v2", run_fusion_v2)]:
    ret = fn()
    acc, use = ret[0], ret[1]
    if name in ("融合制", "融合制v2"):
        marvises = ret[2]
    stats(name, acc, use)
print("-" * 88)
print("v2 各分身工作统计（调用次数 / 提案被采纳数）：")
for m in marvises:
    print(f"  {m.name}({m.role}) : 调用 {m.calls:5d} 次 | 采纳 {m.accepted:6d} 次")
print("-" * 88)
print("AI 思想映射表（分身 ↔ 大模型 MoE 机制）：")
mapping = [
    ("人民/知识体系", "预训练语料分布 / token 流"),
    ("市场部", "路由器 top-k 打分（softmax 竞价的平方级负载惩罚）"),
    ("计划委", "共享专家保底（DeepSeek-V3 式共享 expert）"),
    ("联邦政府", "动态负载均衡 + 紧急扩容（梯度累积触发扩容）"),
    ("议会", "辅助平衡 loss 的超参审议（系数调节）"),
    ("税务局", "auxiliary balance loss + 容量因子 bias（反垄断上限）"),
    ("公益组织", "负载监控器 / 区域灾备路由（gini 均衡监测）"),
    ("教育部", "专家重训 / 知识蒸馏（弱领域知识溢出 0.15）"),
    ("总统", "多门控集成 / final 汇总层"),
    ("特区港", "探索型稀疏专家（免税自由港 = 低 bias 高探索）"),
    ("气象局", "容量规划 / 需求预测（企业制录取：按领域配额动态调度）"),
]
for a, b in mapping:
    print(f"  {a:10s} → {b}")
print("恢复=漂移后正确率回到阶段1的95%所需token; 0=零损失")
