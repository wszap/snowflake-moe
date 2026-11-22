# -*- coding: utf-8 -*-
"""paperplane-Moe V1.003 · 酶：细胞内部常驻的辅助催化层（不是外部调度器）。

为什么重写 V1.002
==================
V1.002 的 EnzymeLineage 是「外部谱系 + 外部打分选择」：
  · select(step, scores) 用 top-k 保留/淘汰  → 那是自然选择的【打分器】，
    不是细胞里自己长出来的酶；
  · observe_substrate(step, "task" if step%2 else "info") 用 step 奇偶造底物
    → 底物不是细胞真实看到的量；
  · catalytic.rate() 乘在 topw 上再 renorm → 改的是【wiring 份额】。
    酶在生物学里降低活化能、提高 kcat，但【不改变反应平衡 Keq，也不决定
    哪个底物被谁选中】；乘份额更像转录因子/调控蛋白，不是酶。

V1.003 按生物学本义重做，四条对应关系
=====================================
  1) 常驻管家（constitutive housekeeping）
     每个专家 slot 出生就带一条组成型酶，浓度恒定、c_i ≡ 1.0。
     酶「没事也在细胞里」辅助日常反应，不是事件触发才存在的临时模组。
  2) 区室化（compartment）
     一个 expert slot = 一个区室，各有一份独立 EnzymePool；线粒体/溶酶体/
     叶绿体各有自己的酶组，不共享全局浓度 → 不产生全局耦合，也就不碰
     clip_grad_norm_（S8/step23 三 seed 证实的 +529,416 参数副作用）。
  3) 催化作用在【专家内部反应】上，不在 wiring 上
         comp_i = topw_i · silu(x·E1_i)·E2_i   →   comp_i · c_i
     c_i 是该区室酶池的有效催化系数 = Vmax_now / Vmax_baseline。
     wiring/topw 一个比特不动 ⇒ 红线 12「不筛选、全权重非零」保住。
  4) 自主进化 = 内源自组织，没有外部 scores
     底物 S_i = 该 slot 真实前向统计量（叶绿体把外部数据转成可用底物）
     米氏-希尔：v_j = kcat_j·[E_j]·S^h/(Km^h + S^h)
     诱导表达：d[E]/dt = k_syn·S/(Ks+S)·atp − k_deg·[E]，含产物抑制负反馈
     分裂     ：基因 duplication → 浓度对半分给子区室 + 特异性/kcat/Km 漂变
                （neofunctionalization 的原型，不是 top-k 打分）
     自噬     ：长期无底物 + ATP 低 → 浓度衰减到阈值以下被回收；
                【只回收酶，绝不删专家】，组成型基线永远保留。

硬前置（任何一条不满足都不许上 A/B）
====================================
  · c_i ≡ 1.0（默认 / 全关）⇒ compose.py 逐位恒等，301 点 loss/eff 全 0 不破
  · c_i > 0（正锥）⇒ 不把「拼」退化成「选」
  · 全部状态是 Python 对象 / float，不进 model.parameters()、不产生梯度
  · 局部 random.Random(seed)，绝不消耗全局 torch RNG（否则数据顺序变）
  · 每次 birth / duplication / autophagy / 状态汇总都写事件，谱系可重放
    （规格 11/13「数据要全 + 观察趋势」）

本模块【不依赖 torch】，因此无 GPU 也能完整自检；真接线由
install_V1.003.sh 对 compose.py / telemetry.py / train_bio.py 打精确锚点补丁。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

ENZ_EVENT_KINDS = (
    "enzyme_birth", "enzyme_duplication", "enzyme_autophagy",
    "enzyme_state", "substrate", "atp_supply", "product_inhibition",
)
ENZ_NUMERIC_FIELDS = (
    "slot_id", "enzyme_id", "lineage_id", "parent_id", "birth_step",
    "kcat", "Km", "specificity", "conc", "vmax", "sat", "substrate_S",
    "catalytic_c", "product_ema", "atp_ratio", "n_alive", "lifespan",
    "mutation_scale", "conc_total", "vmax_total", "inhibition",
)

# 组成型基线：全池唯一必须存在的酶，sat 恒 1（管家酶底物已饱和）
CONSTITUTIVE_KCAT = 1.0
CONSTITUTIVE_CONC = 1.0


@dataclass
class EnzymeSpecies:
    """一条酶（一个基因拷贝 / 一个同工酶）。

    specificity ∈ (0,1]: 对该区室底物的亲和方向；突变会让后代漂变，
    这正是「复制后一个拷贝自由漂变、偶尔获得新底物特异性」的原型。
    """
    enzyme_id: int
    slot_id: int
    lineage_id: int
    parent_id: Optional[int]
    birth_step: int
    specificity: float
    kcat: float
    Km: float
    hill: float
    conc: float
    k_syn: float
    k_deg: float
    constitutive: bool = False
    alive: bool = True
    no_substrate_streak: int = 0

    @property
    def vmax(self) -> float:
        return self.kcat * self.conc


class EnzymePool:
    """一个 expert slot = 一个区室 = 一份独立酶池（区室化，不共享）。"""

    def __init__(self, slot_id: int, rng: random.Random,
                 k_syn: float = 0.06, k_deg: float = 0.02,
                 Ks: float = 0.35, Ki: float = 1.2,
                 eps: float = 1e-3, patience: int = 60,
                 beta_ema: float = 0.95, dt: float = 1.0,
                 induce_at: float = 0.30, induce_streak: int = 40,
                 max_induced: int = 3):
        self.slot_id = int(slot_id)
        self.rng = rng
        self.k_syn, self.k_deg = float(k_syn), float(k_deg)
        self.Ks, self.Ki = float(Ks), float(Ki)
        self.eps, self.patience = float(eps), int(patience)
        self.beta_ema, self.dt = float(beta_ema), float(dt)
        # 底物持续出现 → 诱导基因表达（lac-operon 式，不是 hits==N 硬规则）
        self.induce_at = float(induce_at)
        self.induce_streak = int(induce_streak)
        self.max_induced = int(max_induced)
        self.high_streak = 0
        self.product_ema = 0.0
        self._last_c = 1.0                 # 前向读取的催化系数缓存
        self.species: List[EnzymeSpecies] = []
        self._next_id = 0
        self._baseline_vmax = CONSTITUTIVE_KCAT * CONSTITUTIVE_CONC
        self._events: List[dict] = []
        self._spawn_constitutive()

    # ------------------------------------------------------------ 出生
    def _new(self, step: int, specificity: float, kcat: float, Km: float,
             conc: float, lineage: Optional[int], parent: Optional[int],
             constitutive: bool, mutation: float = 0.0) -> EnzymeSpecies:
        eid = self._next_id
        self._next_id += 1
        e = EnzymeSpecies(
            enzyme_id=eid, slot_id=self.slot_id,
            lineage_id=int(lineage if lineage is not None else eid),
            parent_id=parent, birth_step=int(step),
            specificity=float(min(1.0, max(1e-3, specificity))),
            kcat=float(max(1e-3, kcat)), Km=float(max(1e-3, Km)),
            hill=2.0, conc=float(max(0.0, conc)),
            k_syn=self.k_syn, k_deg=self.k_deg,
            constitutive=bool(constitutive))
        self.species.append(e)
        self._events.append(dict(
            step=int(step), kind="enzyme_birth", event="enzyme_birth",
            slot_id=self.slot_id, enzyme_id=eid, lineage_id=e.lineage_id,
            parent_id=parent, birth_step=int(step), kcat=round(e.kcat, 6),
            Km=round(e.Km, 6), specificity=round(e.specificity, 6),
            conc=round(e.conc, 6), mutation_scale=float(mutation),
            constitutive=bool(constitutive)))
        return e

    def _spawn_constitutive(self):
        self._new(0, specificity=1.0, kcat=CONSTITUTIVE_KCAT, Km=1.0,
                  conc=CONSTITUTIVE_CONC, lineage=0, parent=None,
                  constitutive=True)

    # ------------------------------------------------------------ 动力学
    def _sat(self, S: float, e: EnzymeSpecies) -> float:
        """米氏-希尔饱和；组成型恒 1（管家酶底物已饱和）。"""
        if e.constitutive:
            return 1.0
        s = max(0.0, float(S)) * e.specificity
        sh = s ** e.hill
        return sh / (e.Km ** e.hill + sh + 1e-12)

    def observe(self, step: int, S: float, atp_ratio: float,
                emit_state: bool = False) -> float:
        """一步内源动力学：底物 → 表达 → 催化 → 产物抑制 → 自噬。

        返回该区室本步的有效催化系数 c_i（>0）；无诱导酶时精确 = 1.0。
        """
        atp_ratio = min(1.0, max(0.0, float(atp_ratio)))
        S = max(0.0, float(S))
        inhibition = 1.0 / (1.0 + self.product_ema / self.Ki)
        v_total = 0.0

        # ---- 底物持续出现 → 细胞自己表达一条诱导酶（lac-operon 式）----
        # 不是外部 select(scores)、也不是 step%2 造出来的假底物；
        # 高底物连续 induce_streak 步才表达，且受 max_induced 上限约束。
        self.high_streak = (self.high_streak + 1
                            if S > self.induce_at else 0)
        n_induced = sum(1 for e in self.species
                        if e.alive and not e.constitutive)
        if (self.high_streak >= self.induce_streak
                and self.high_streak % self.induce_streak == 0
                and n_induced < self.max_induced):
            self._new(
                step,
                specificity=1.0 + self.rng.uniform(-0.10, 0.10),
                kcat=CONSTITUTIVE_KCAT * (1.0 + self.rng.uniform(-0.05, 0.05)),
                Km=1.0 + self.rng.uniform(-0.10, 0.10),
                conc=0.01,                      # 新表达，浓度从很低起步
                lineage=None, parent=None, constitutive=False,
                mutation=0.0)

        for e in self.species:
            if not e.alive:
                continue
            sat = self._sat(S, e)
            v = e.vmax * sat
            v_total += v
            if e.constitutive:
                continue
            # 诱导表达：合成受底物、ATP、产物抑制三重门控；降解恒在
            induce = S / (self.Ks + S + 1e-12)
            dE = (self.k_syn * induce * atp_ratio * inhibition
                  - self.k_deg * e.conc) * self.dt
            e.conc = max(0.0, e.conc + dE)
            # 内源自噬：长期无底物 + 浓度衰减 → 回收【酶】，不碰专家
            e.no_substrate_streak = (e.no_substrate_streak + 1
                                     if S * e.specificity < 1e-3 else 0)
            if (e.conc < self.eps
                    and e.no_substrate_streak > self.patience):
                e.alive = False
                self._events.append(dict(
                    step=int(step), kind="enzyme_autophagy",
                    event="enzyme_autophagy", slot_id=self.slot_id,
                    enzyme_id=e.enzyme_id, lineage_id=e.lineage_id,
                    lifespan=int(step - e.birth_step),
                    conc=round(e.conc, 6)))

        self.product_ema = (self.beta_ema * self.product_ema
                            + (1.0 - self.beta_ema) * v_total)
        c = v_total / self._baseline_vmax
        c = min(2.0, max(0.5, c))                      # 正锥硬界，不退化成选
        self._last_c = c

        if emit_state:
            self._events.append(dict(
                step=int(step), kind="enzyme_state", event="enzyme_state",
                slot_id=self.slot_id, n_alive=sum(e.alive for e in self.species),
                conc_total=round(sum(e.conc for e in self.species if e.alive), 6),
                vmax_total=round(v_total, 6), catalytic_c=round(c, 6),
                substrate_S=round(S, 6), product_ema=round(self.product_ema, 6),
                inhibition=round(inhibition, 6), atp_ratio=round(atp_ratio, 6)))
        return c

    # ------------------------------------------------------------ 自主进化
    def duplicate(self, step: int, mutation: float = 0.08) -> Optional[dict]:
        """细胞分裂时的基因 duplication：浓度对半分 + 特异性/kcat/Km 漂变。

        没有外部 scores、没有 top-k。选「当前浓度最高的存活酶」是细胞自己
        的表达量结果，不是外部适应度打分。
        """
        # 只复制【非组成型】；组成型管家酶浓度恒定，DNA 复制不减已有蛋白
        inducible = [e for e in self.species
                     if e.alive and not e.constitutive and e.conc > self.eps]
        if inducible:
            src = max(inducible, key=lambda e: (e.conc, -e.enzyme_id))
            src.conc *= 0.5                          # 细胞质分配给子区室
            child_conc = src.conc
        else:
            # 只有管家酶时：复制它的基因，新拷贝自由漂变（neofunctionalization），
            # 母区室保留原浓度 ⇒ baseline 恒 1.0，c 可精确回到 1.0
            src = next(e for e in self.species if e.constitutive)
            child_conc = 0.05
        child = self._new(
            step,
            specificity=src.specificity * (1.0 + self.rng.uniform(-mutation, mutation)),
            kcat=src.kcat * (1.0 + self.rng.uniform(-mutation, mutation)),
            Km=src.Km * (1.0 + self.rng.uniform(-mutation, mutation)),
            conc=child_conc,
            lineage=src.lineage_id, parent=src.enzyme_id,
            constitutive=False, mutation=mutation)
        self._events.append(dict(
            step=int(step), kind="enzyme_duplication",
            event="enzyme_duplication", slot_id=self.slot_id,
            enzyme_id=child.enzyme_id, parent_id=src.enzyme_id,
            lineage_id=child.lineage_id, kcat=round(child.kcat, 6),
            Km=round(child.Km, 6), specificity=round(child.specificity, 6),
            conc=round(child.conc, 6), mutation_scale=float(mutation)))
        return {"child": child.enzyme_id, "parent": src.enzyme_id}

    def catalytic_factor(self) -> float:
        """不推进时间，只返回上一步动力学算出的 c_i（供前向读取；基准恒 1.0）。"""
        return float(self._last_c)

    def summary(self) -> dict:
        return dict(slot_id=self.slot_id,
                    n_alive=sum(e.alive for e in self.species),
                    n_total=len(self.species),
                    c=self.catalytic_factor(),
                    product_ema=self.product_ema)


class Enzymology:
    """全细胞酶学协调器：叶绿体 → 线粒体 → 各区室酶池 → 事件。

    · 叶绿体 chloroplast：把外部输入统计量转成可用底物 S（纯数值，不改 x）
    · 线粒体 mitochondria：ATP 供给 atp_ratio，门控 k_syn（合成要耗能）
    · 血液两类忆点：info/task 底物比在此汇合（blood_info / blood_task）
    · 小核架 / 稀疏回顾：product_ema + conc 是无参数的长时程状态
    """

    def __init__(self, n_slots: int = 16, seed: int = 0,
                 enabled: bool = False, mutation: float = 0.08,
                 state_every: int = 500):
        self.n_slots = int(n_slots)
        self.enabled = bool(enabled)
        self.mutation = float(mutation)
        self.state_every = int(state_every)
        self.rng = random.Random(int(seed) * 104729 + 613)
        self.pools: List[EnzymePool] = [
            EnzymePool(i, random.Random(int(seed) * 104729 + 613 + i))
            for i in range(self.n_slots)]
        self.events: List[dict] = []
        self._flush_cursor = 0
        self.atp_ratio = 1.0
        if not self.enabled:
            # 组成型酶【存在】（谱系可重放），但 OFF 档不写任何事件：
            # 默认档 events 与历史 json 逐位一致（规格 11/13 零污染）
            for p in self.pools:
                p._events.clear()

    # ------------------------------------------------------------ 每步
    def observe(self, step: int, obs: Sequence[float],
                atp_ratio: float = 1.0, blood_info: float = 0.0,
                blood_task: float = 0.0, emit_state: bool = False) -> List[float]:
        """obs[i] = 第 i 个专家 slot 本步真实前向统计量（底物浓度）。

        返回长度 n_slots 的催化系数列表；enabled=False 时精确返回全 1.0，
        且不产生任何事件（默认档 events 逐位不变）。
        """
        ones = [1.0] * self.n_slots
        if not self.enabled:
            return ones
        self.atp_ratio = min(1.0, max(0.0, float(atp_ratio)))
        o = [float(v) for v in obs] + [0.0] * self.n_slots
        bi, bt = float(blood_info), float(blood_task)
        factors: List[float] = []
        log = bool(emit_state or step % self.state_every == 0)
        for i, pool in enumerate(self.pools):
            # 叶绿体：外部统计量 + 两类忆点 → 该区室可用底物 S
            S = max(0.0, o[i]) * (1.0 + 0.1 * (bi + bt))
            c = pool.observe(step, S, self.atp_ratio, emit_state=log)
            factors.append(c)
            if log:
                pool._events.append(dict(
                    step=int(step), kind="substrate", event="substrate",
                    slot_id=i, substrate_S=round(S, 6),
                    atp_ratio=round(self.atp_ratio, 6),
                    inhibition=round(
                        1.0 / (1.0 + pool.product_ema / pool.Ki), 6)))
        if log:
            self.events.append(dict(
                step=int(step), kind="atp_supply", event="atp_supply",
                atp_ratio=round(self.atp_ratio, 6),
                product_inhibition=round(
                    sum(p.product_ema for p in self.pools) / self.n_slots, 6)))
        return factors

    def factors(self) -> List[float]:
        if not self.enabled:
            return [1.0] * self.n_slots
        return [p.catalytic_factor() for p in self.pools]

    # ------------------------------------------------------------ 分裂
    def on_divide(self, step: int, parent_slot_ids: Sequence[int]) -> int:
        """规格 2/3：分裂时自主传承 —— 酶池 duplication，不靠外部 scores。"""
        if not self.enabled:
            return 0
        n = 0
        for sid in parent_slot_ids:
            pool = self.pools[int(sid) % self.n_slots]
            if pool.duplicate(step, self.mutation) is not None:
                n += 1
        return n

    # ------------------------------------------------------------ 观测 / 重放
    def collect_events(self) -> List[dict]:
        out: List[dict] = []
        for p in self.pools:
            out.extend(p._events)
            p._events.clear()
        out.extend(self.events)
        self.events.clear()
        return out

    def flush(self, rec, step: Optional[int] = None) -> int:
        """游标式灌进 TelemetryRecorder（不清空全局 ⇒ 谱系可完整重放）。"""
        n = 0
        for ev in self.collect_events():
            fields = {k: v for k, v in ev.items()
                      if k not in ("kind", "step", "event")}
            rec.emit(ev["kind"], int(ev.get("step", step or 0)), **fields)
            n += 1
        return n

    def replay(self) -> dict:
        """规格 11/13：谱系可重放；所有酶有 lineage_id，消亡带 lifespan。"""
        all_e = [e for p in self.pools for e in p.species]
        return dict(
            n_slots=self.n_slots,
            total_species=len(all_e),
            alive=sum(e.alive for e in all_e),
            constitutive=sum(e.constitutive for e in all_e),
            lineages=len(set(e.lineage_id for e in all_e)),
            with_parent=sum(e.parent_id is not None for e in all_e),
            all_have_lineage=all(e.lineage_id is not None for e in all_e),
            baseline_intact=all(
                any(e.constitutive and e.alive for e in p.species)
                for p in self.pools),
        )

    def summary(self) -> dict:
        cs = [p.catalytic_factor() for p in self.pools]
        return dict(n_slots=self.n_slots,
                    c_min=min(cs), c_max=max(cs),
                    c_mean=sum(cs) / len(cs),
                    n_alive=sum(s["n_alive"] for s in
                                (p.summary() for p in self.pools)),
                    atp_ratio=self.atp_ratio)


# ---------------------------------------------------------------- 自检
def verify_enzymology(seed: int = 0) -> tuple:
    """纯 Python 自检：只验「不破坏 + 动力学性质」，不验收益。"""
    off = Enzymology(n_slots=8, seed=seed, enabled=False)
    off_id = off.observe(0, [0.5] * 8) == [1.0] * 8
    off_no_events = len(off.collect_events()) == 0
    off_factors = off.factors() == [1.0] * 8

    on = Enzymology(n_slots=8, seed=seed, enabled=True, state_every=500)
    # 前 300 步只给 2 个 slot 持续底物（其余无底物）
    hist_c = []
    for s in range(0, 600):
        obs = [0.9 if i in (1, 5) else 0.0 for i in range(8)]
        cs = on.observe(s, obs, atp_ratio=0.9, blood_info=0.2, blood_task=0.3)
        hist_c.append(cs)
    base = [p.catalytic_factor() for p in on.pools]
    cone = all(0.5 <= c <= 2.0 for c in base)
    # 有底物的 slot 应出现诱导酶（c>1 或至少 n_alive>1）
    induced = any(p.summary()["n_alive"] > 1 for p in
                  (on.pools[1], on.pools[5]))
    # 无底物 slot 不应凭空长出诱导酶（内源，非 step 规则）
    quiet = all(p.summary()["n_alive"] == 1 for p in
                (on.pools[0], on.pools[2], on.pools[3]))
    dup = on.on_divide(600, [1, 5, 1, 5])
    with_parent = on.replay()["with_parent"] >= 2
    baseline_intact = on.replay()["baseline_intact"]
    # 分裂后继续无底物 → 诱导酶被自噬回收，c 回到 1.0 附近
    for s in range(600, 1400):
        on.observe(s, [0.0] * 8, atp_ratio=0.1)
    decayed = all(abs(p.catalytic_factor() - 1.0) < 1e-6
                  for p in on.pools)
    evs = on.collect_events()
    kinds = set(e["kind"] for e in evs)
    autophagy_written = "enzyme_autophagy" in kinds
    lifespan_ok = all("lifespan" in e for e in evs
                      if e["kind"] == "enzyme_autophagy")
    rep = on.replay()

    checks = [
        ("★ OFF：factors 全 1.0（301 点零污染硬前置）", off_factors and off_id),
        ("★ OFF：不写任何事件（默认档 events 逐位不变）", off_no_events),
        ("★ 催化系数恒在正锥 [0.5,2.0]，不退化成选", cone),
        ("底物重复 → 该区室诱导酶（内源，非 step 规则）", induced),
        ("无底物区室不长诱导酶", quiet),
        ("分裂 duplication ≥2，后代带 parent_id", dup >= 2 and with_parent),
        ("组成型管家酶每个区室永远保留", baseline_intact),
        ("长期无底物 + ATP 低 → c 回到 1.0（可逆自噬）", decayed),
        ("自噬事件带 lifespan（规格 11/13）", autophagy_written and lifespan_ok),
        ("事件含 birth/duplication/state/substrate/atp",
         all(k in kinds for k in ("enzyme_birth", "enzyme_duplication",
                                  "enzyme_state", "substrate", "atp_supply"))),
        ("谱系可重放：全部酶有 lineage_id", rep["all_have_lineage"]),
    ]
    return checks, dict(replay=rep, c_after=base, n_events=len(evs),
                        kinds=sorted(kinds))


if __name__ == "__main__":
    checks, info = verify_enzymology()
    for name, ok in checks:
        print("  %s  %s" % ("PASS" if ok else "FAIL", name))
    print("  replay=%s" % info["replay"])
    print("  c_after=%s" % [round(c, 6) for c in info["c_after"]])
    print("  events=%d kinds=%s" % (info["n_events"], info["kinds"]))
