# -*- coding: utf-8 -*-
"""step29c 判据：评分表从「固定步数终态 val」换成 τ / T_ε / 双线成本。

纯标准库，只读 json，不跑训练。判读口径跑之前写死，不许事后挑数。

================================================================================
本版相对 step29 原版只改 [0]（step29c 修正）
================================================================================
原版 [0] 写死 `REF=st25_s0_B`（3000 步归档），拿 20 / 300 步档去逐位比。
但 `train_bio.py:226` 的 `warm = max(1, max_steps // 10)` 让 lr 调度随
max_steps 变（20 步 warm=2 / 300 步 warm=30 / 3000 步 warm=300），
loss/eff 必然不同 ⇒ 这是【参考系选错】，不是补丁破坏零污染。
step29b 已把 `step29_smoke.py` 的 U3 修好，但本文件里还留着同一份旧逻辑，
会在 SHORT 关误报「FAIL → 回滚停线」。本次同步修掉。

  [0] 新逻辑：
      ① 优先找 `output/v4/st29_ref*_nopatch.json`（train_bio.py.bak_step29
         跑的同 max_steps 未补丁基线；它 import 补丁版 compose.py ⇒ 同时实证
         那两行 detach 统计不改数值）。共有字段逐位全 0 ⇒ PASS。
      ② 没找到 ⇒ 回退归档 REF；max_steps 不同 ⇒ SKIP 并打印理由，
         **不判 FAIL、也不假装 PASS**，真 301 点硬前置留给 FULL 关 --zero-check。
      ③ 只有「同配置基线」出现差值（哪怕 1 ULP）才判 FAIL 停线。

另加软防护：若当前 tags 全是第一关 `_s20`（SHORT 还没落盘），
[2]~[6] 标「仅预览，不作结论」，避免「判据跑早了」被当成结果。

  [1] 三条单元测试：T1 f 动态 / T2 e 变化 / T3 corr(k,1/H)
  [2] 主判据  T_ε：固定【绝对】loss 阈值（默认 1.30，env STEP29_THR 可改）
      的插值到达步数。不用「各档末态×系数」——那会测成第一步瞬态。
  [3] 渐近收敛率：step≥1000 段 log(loss−floor) 线性拟合的 −slope
  [4] 承重对照：cat vs const / lrx（排除「只是等效调大 lr」）、vs amp_old
  [5] κ_eff 代理：spread(k·H) 应小于 spread(H)（酶压缩局部曲率离散度）
  [6] 双线成本：到 T_ε 的总算力 ∝ T + 累计酶成本
  [7] 速率 rescue：cat_freeze（单 cell k≡1）与 cat 的差

  ★ 单 seed：所有差值只作方向，不叫显著；终态 val 降级为「不劣于」门禁。
  ★ 永远不写「加速」：真 wall-clock / avg_step 三档对比不在本判据内。
"""
import glob
import json
import math
import os
import statistics
import sys

V4 = os.environ.get("V4DIR", "output/v4")
THR = float(os.environ.get("STEP29_THR", "1.30"))
REF = os.environ.get("STEP29_REF", "st25_s0_B")
CMP_FIELDS = ("loss", "eff", "lr", "var_norm", "clip_main")


def L(tag):
    p = os.path.join(V4, tag + ".json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def reach(hist, thr):
    for i in range(1, len(hist)):
        a, b = hist[i - 1], hist[i]
        if b["loss"] <= thr:
            if a["loss"] <= thr:
                return float(a["step"])
            f = (a["loss"] - thr) / max(a["loss"] - b["loss"], 1e-12)
            return float(a["step"]) + f * (b["step"] - a["step"])
    return None


def asym_slope(hist, floor):
    pts = [(h["step"], h["loss"] - floor) for h in hist
           if h["step"] >= 1000 and h["loss"] - floor > 1e-9]
    if len(pts) < 8:
        return None
    xs = [p[0] for p in pts]
    ys = [math.log(p[1]) for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    if vx <= 0:
        return None
    return -sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx


def spread(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    if len(v) < 2:
        return 0.0, 0.0, 0.0
    return min(v), max(v), (max(v) / max(min(v), 1e-12))


def mean(v):
    q = [x for x in v if x is not None and math.isfinite(x)]
    return statistics.mean(q) if q else float("nan")


def _max_steps(d):
    c = (d or {}).get("config") or {}
    return int(c.get("max_steps", 0)) or None


def _strict_compare(A, R):
    ha = {h["step"]: h for h in R.get("history", [])}
    hb = {h["step"]: h for h in A.get("history", [])}
    cs = sorted(set(ha) & set(hb))
    if not cs:
        return False, dict(reason="无共同 history 点")
    fields = [k for k in CMP_FIELDS
              if all(k in ha[s] and k in hb[s] for s in cs)]
    if not fields:
        return False, dict(reason="无共有可比字段")
    mx = {k: max(abs(float(ha[s][k]) - float(hb[s][k])) for s in cs)
          for k in fields}
    ok = all(v == 0.0 for v in mx.values())
    dl = mx.get("loss", float("nan"))
    de = mx.get("eff", float("nan"))
    return ok, dict(points=len(cs), fields=fields, dl=dl, de=de,
                    **{("max_d" + k): v for k, v in mx.items()})


def hard_gate(A0):
    """返回 (state, info)：state ∈ {True=PASS, False=FAIL停线, None=SKIP}。"""
    ms = _max_steps(A0)
    # ① 同配置未补丁基线优先
    cands = []
    for p in sorted(glob.glob(os.path.join(V4, "st29_ref*_nopatch.json"))):
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if _max_steps(r) == ms:
            cands.append((os.path.basename(p), r))
    if cands:
        name, r = cands[0]
        ok, info = _strict_compare(A0, r)
        return (True if ok else False), dict(
            ref=name, kind="同配置未补丁基线", max_steps=ms, **info)
    # ② 回退归档；配置不同 ⇒ SKIP（不判失败、不假装通过）
    R = L(REF)
    if not R:
        return None, dict(reason="缺同配置基线，也缺 %s" % REF)
    rms = _max_steps(R)
    if ms and rms and ms != rms:
        return None, dict(
            reason="参考系不可比：本档 max_steps=%s，归档 %s max_steps=%s"
                   "（warm=max_steps//10 ⇒ lr 调度不同）；"
                   "真 301 点硬前置留给 FULL 关 --zero-check" % (ms, REF, rms),
            ref=REF, kind="归档（不同配置）")
    ok, info = _strict_compare(A0, R)
    return (True if ok else False), dict(ref=REF, kind="归档（同配置）", **info)


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not tags:
        p = "step29_tags.txt"
        if os.path.exists(p):
            tags = open(p, encoding="utf-8").read().split()
        else:
            tags = ["st29_A0_B", "st29_A1_res", "st29_amp_old", "st29_cat",
                    "st29_cat_het", "st29_const", "st29_lrx", "st29_cat_freeze"]
    D = {t: L(t) for t in tags}
    missing = [t for t, d in D.items() if d is None]
    print("=" * 84)
    print("step29c 判据：酶=速率常数 / 主指标 T_ε·渐近率·双线成本（终态 val 只作门禁）")
    print("V4DIR=%s  THR=%.4f  REF=%s  档数=%d" % (V4, THR, REF, len(D)))
    if missing:
        print("  缺档：%s（不是「还没跑完」，是这些档没落盘；不猜）" % missing)
    # 软防护：判据读到第一关 20 步 json
    preview = bool(D) and all(
        (_max_steps(d) or 0) < 100 for d in D.values() if d)
    if preview:
        print("  ⚠ 当前 tags 全是第一关 _s20（SHORT 还没落盘）→ [2]~[6] 仅预览，不作结论")
    print("=" * 84)

    # ---------------- [0] 硬前置 ----------------
    A0 = next((D[t] for t in D if "A0_B" in t), None)
    hard = None
    print("\n[0] 零污染硬前置（同配置基线；配置不同只 SKIP，不误判停线）")
    if A0 is None:
        print("  缺 A0_B → FAIL")
        hard = False
    else:
        state, info = hard_gate(A0)
        hard = state
        tr = len(A0.get("enz_rate_trace", []))
        p0, p1 = A0.get("params_start"), A0.get("params_end")
        print("  off 档 enz_rate_trace 长度=%d（应为 0）  params %s→%s"
              % (tr, p0, p1))
        if state is None:
            print("  [U3] %s" % info)
            print("  → SKIP（参考系不可比，不是训练回归）→ 可往下读，"
                  "真 301 点硬前置仍在 FULL 关 --zero-check")
        else:
            print("  [U3] %s" % info)
            print("  对齐点=%s  loss max|Δ|=%.3e  eff max|Δ|=%.3e"
                  % (info.get("points"), info.get("dl", float("nan")),
                     info.get("de", float("nan"))))
            if state is True:
                print("  → PASS → 地基干净，可往下读")
            else:
                print("  → FAIL → 同配置基线出现差值，真零污染被破：回滚 *.bak_step29，停线")

    # ---------------- [1] 三条单元测试 ----------------
    print("\n[1] 三条单元测试（T3 只要求 cat / cat_het）")
    units = {}
    for t, d in D.items():
        if not d:
            continue
        m = d.get("enz_rate_mode", "off")
        if m not in ("cat", "cat_het", "const", "lrx") or "freeze" in t:
            continue
        s = d.get("enz_rate_summary") or {}
        tr = d.get("enz_rate_trace", [])
        f_all = [v for q in tr for v in q.get("f", [])]
        T1 = (statistics.pstdev(f_all) > 0.05 if f_all else False) and \
             (0.05 < statistics.median(f_all) < 0.95 if f_all else False)
        T2 = s.get("e_delta_mean", 0) > 0.10
        T3 = s.get("corr_k_invH_mean", 0) > 0.5
        need = ("T1", "T2", "T3") if m in ("cat", "cat_het") else ("T1", "T2")
        ok = all({"T1": T1, "T2": T2, "T3": T3}[k] for k in need)
        units[t] = ok
        print("  %-22s f_std=%.3f eΔ=%.3f corr=%.3f  需过%s → %s"
              % (t, statistics.pstdev(f_all) if f_all else 0,
                 s.get("e_delta_mean", 0), s.get("corr_k_invH_mean", 0),
                 need, "PASS" if ok else "FAIL"))

    # ---------------- [2] 主判据 ----------------
    floors = [min(h["loss"] for h in d["history"]) for d in D.values() if d]
    floor = min(floors) - 1e-4 if floors else 0.0
    print("\n[2] 主判据（阈值固定绝对 loss=%.4f；floor=%.6f 供渐近率）%s"
          % (THR, floor, "  【仅预览】" if preview else ""))
    T, SLP, COST = {}, {}, {}
    print("  %-24s %10s %12s %10s" % ("tag", "T_ε", "渐近率", "终态门禁"))
    for t, d in D.items():
        if not d:
            continue
        h = d["history"]
        T[t] = reach(h, THR)
        SLP[t] = asym_slope(h, floor)
        tr = d.get("enz_rate_trace", [])
        if T[t] is not None and tr:
            i = min(int(T[t]), len(tr) - 1)
            COST[t] = tr[i].get("cost_sum", 0.0)
        else:
            COST[t] = 0.0
        gate = "PASS" if d["val_loss"] <= (A0["val_loss"] * 1.05
                                           if A0 else 9e9) else "FAIL"
        print("  %-24s %10s %12s %10s"
              % (t, "None" if T[t] is None else "%.1f" % T[t],
                 "None" if SLP[t] is None else "%.5f" % SLP[t], gate))

    def pair(a, b):
        if a is None or b is None:
            return float("nan")
        return a - b

    def find(suffix):
        key = "st29_" + suffix
        exact = [t for t in D if t == key]
        if exact:
            return exact[0]
        alt = [t for t in D if t.startswith(key + "_s")]
        return alt[0] if alt else None

    c = find("cat")
    co = find("const")
    lx = find("lrx")
    am = find("amp_old")
    rs = find("A1_res")
    fr = find("cat_freeze")
    ht = find("cat_het")

    print("\n[3] 承重对照（负=更早；cat 必须同时赢过两个等效 lr 对照）")
    for nm, ref in [("cat−A1_res", rs), ("cat−const", co),
                    ("cat−lrx", lx), ("cat−amp_old", am)]:
        x = T.get(c) if c else None
        y = T.get(ref) if ref else None
        dv = pair(x, y)
        print("  %-14s = %s   (%s vs %s)"
              % (nm, "%+8.1f" % dv if dv == dv else "     n/a", c, ref))
    p3 = (not preview and c is not None and co and lx
          and T.get(c) is not None
          and T.get(c) < 0.8 * mean([T.get(co), T.get(lx)])
          and pair(T.get(c), T.get(co)) < -10
          and pair(T.get(c), T.get(lx)) < -10)
    print("  → 与等效 lr 严格可分: %s%s"
          % ("PASS" if p3 else "FAIL", "  【仅预览】" if preview else ""))

    print("\n[4] κ_eff 代理：spread(k·H) 应小于 spread(H)（酶压平局部曲率）")
    for t in (c, ht, co, lx):
        if not t or not D.get(t):
            continue
        tr = D[t].get("enz_rate_trace", [])
        if not tr:
            continue
        q = tr[-1]
        H, k = q.get("H", []), q.get("k", [])
        kH = [a * b for a, b in zip(k, H)]
        _, _, rH = spread(H)
        _, _, rkH = spread(kH)
        print("  %-22s κ(H)=%8.2f  κ(k·H)=%8.2f  比=%.3f"
              % (t, rH, rkH, rkH / max(rH, 1e-12)))

    print("\n[5] 双线成本（到 T_ε 的总算力 ∝ T + 累计酶成本；基线=A1_res）")
    base = (T[rs] + COST[rs]) if (rs and T.get(rs) is not None) else None
    for t in D:
        if T.get(t) is None:
            print("  %-24s T_ε=None（未达阈值，不计成本）" % t)
            continue
        tot = T[t] + COST[t]
        print("  %-24s T=%8.1f  cost=%8.2f  total=%8.1f  %s"
              % (t, T[t], COST[t], tot,
                 "" if base is None else "相对基线=%.3f×" % (tot / max(base, 1e-9))))

    print("\n[6] 速率 rescue（cat_freeze：单 cell k≡1，不删专家、不改前向）")
    if fr and c:
        print("  cat T_ε=%s   freeze T_ε=%s   Δ=%+.1f"
              % (T.get(c), T.get(fr), pair(T.get(fr), T.get(c))))
        print("  （freeze 变慢 = 该 cell 的酶确实在贡献速率；仅描述，不设阈值）")
    else:
        print("  未跑 freeze 档（第三关才有）")

    print("\n【判读 · 预先写死】")
    print("  [0] 同配置基线 FAIL → 回滚 *.bak_step29，停线，别读下面任何一行")
    print("  [0] SKIP → 只是参考系不可比，不是停线；真 301 点留给 FULL 关 --zero-check")
    if preview:
        print("  ⚠ tags 全是 _s20 → 判据跑早了，[2]~[6] 仅预览，等 SHORT 落盘再重跑")
    print("  [1] 任一 cat 档不过 → 酶根本没上场，结论只能写『实验未测到』")
    print("  [3] PASS 且 T_ε(cat) 明显更小 → 速率常数型酶在拼专家里成立")
    print("  [3] FAIL / T_ε 打平 → 与等效 lr 不可分，如实收口，不调 β/Km 硬凑")
    print("  单 seed 仅方向，不叫显著；终态 val 只是门禁，不作排序目标")
    print("  永远不写『加速』：wall-clock / avg_step 三档对比不在本判据内")
    print("=" * 84)


if __name__ == "__main__":
    main()
