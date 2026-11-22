# -*- coding: utf-8 -*-
"""step29 第一关：20 步 smoke 的三条单元测试 + 零污染哨兵。

纯标准库，只读 json，不跑训练、不改任何文件。
判定全部预先写死：任一不过 ⇒ SMOKE_OK=0 ⇒ 不许起 300 步 / 3000 步。

  T1  Km 标定未让 f 退化（活跃窗口 std(f)>0.05 且 0.05<median(f)<0.95）
      —— MM 本来就有 S≪Km 的一级区(f→0) 与 S≫Km 的零级区(f→1)，
         禁止 f 触及两端反而违反 MM；只验「没退化成常数」。
  T2  诱导慢变量真变化（mean|e_end−e_start| > 0.10）
  T3  酶压在瓶颈上（后半段 mean corr(k, 1/H) > 0.5）
  U2  off 档 enz_rate_trace 为空、k 全 1
  U3  ★修正版★ 零污染哨兵改为【同配置未补丁基线】逐位比对

================================================================================
为什么改 U3（step29b）
================================================================================
原版拿 20 步档去逐位比 `st25_s0_B.json`（3000 步归档）。但
`train_bio.py:226` 的 `warm = max(1, max_steps // 10)` 让学习率调度随
max_steps 变：20 步档 warm=2、3000 步档 warm=300，到 step10 时
lr 分别是 3.0e-5 与 2.99e-4（差 10×）⇒ loss/eff 必然不同。
那是【参考系选错】，不是补丁破坏零污染。

修正：
  1. 优先找 `output/v4/st29_ref*_nopatch.json`（用 train_bio.py.bak_step29
     跑的同 max_steps 未补丁基线）。它运行时用的 compose.py 是补丁版
     （多两行 detach 统计）⇒ 这一步同时实证那两行不改数值。
  2. 找到且 max_steps 相同 ⇒ 比 loss/eff/lr/var_norm/clip_main 共有字段逐位。
     全 0 才 PASS；差一个 ULP 就是真零污染被破 ⇒ FAIL 停线。
  3. 没找到 ⇒ 才回退归档 `st25_s0_B.json`；若 max_steps 不同 ⇒ SKIP
     并打印理由（不假装通过、也不误判失败）。
  4. **真 301 点硬前置留给 FULL 关的 `--zero-check`**（warmup 一致才有意义）。

用法：
  python3 step29_smoke.py --self-test
  python3 step29_smoke.py st29_A0_B_s20 st29_cat_s20 st29_const_s20 st29_lrx_s20
"""
import glob
import json
import math
import os
import statistics
import sys

V4 = os.environ.get("V4DIR", "output/v4")
CMP_FIELDS = ("loss", "eff", "lr", "var_norm", "clip_main")


def L(tag):
    p = os.path.join(V4, tag + ".json")
    if not os.path.exists(p):
        raise SystemExit("缺 %s → smoke 还没跑完" % p)
    return json.load(open(p, encoding="utf-8"))


def corr(xs, ys):
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def unit_checks(d, label):
    """返回 (T1,T2,T3, 详细 dict)"""
    tr = d.get("enz_rate_trace", [])
    sm = d.get("enz_rate_summary") or {}
    steps = len(tr)
    if steps < 2:
        return False, False, False, dict(reason="trace 为空或只有 1 步", steps=steps)

    f_all = [v for t in tr for v in t["f"]]
    f_std = statistics.pstdev(f_all) if f_all else 0.0
    f_med = statistics.median(f_all) if f_all else 0.0
    T1 = (f_std > 0.05) and (0.05 < f_med < 0.95)

    e0, e1 = tr[0]["e"], tr[-1]["e"]
    e_delta = sum(abs(a - b) for a, b in zip(e0, e1)) / max(len(e0), 1)
    T2 = e_delta > 0.10

    cs = [t["corr_k_invH"] for t in tr[steps // 2:]]
    c_mean = statistics.mean(cs) if cs else 0.0
    k_last, H_last = tr[-1]["k"], tr[-1]["H"]
    c_recalc = corr(k_last, [1.0 / max(h, 1e-12) for h in H_last])
    T3 = (c_mean > 0.5) and (c_recalc > 0.3)

    return T1, T2, T3, dict(
        steps=steps, f_std=round(f_std, 4), f_med=round(f_med, 4),
        f_min=round(min(f_all), 4) if f_all else None,
        f_max=round(max(f_all), 4) if f_all else None,
        e_delta=round(e_delta, 4),
        corr_mean=round(c_mean, 4), corr_recalc=round(c_recalc, 4),
        k_last_mean=round(tr[-1]["k_mean"], 4),
        k_last_std=round(tr[-1]["k_std"], 4),
        H_min=tr[-1]["H_min"], H_max=tr[-1]["H_max"],
        summary_corr=sm.get("corr_k_invH_mean"),
    )


def off_check(d, label):
    tr = d.get("enz_rate_trace", [])
    mode = d.get("enz_rate_mode", "off")
    hist = d.get("history", [])
    ks = [h.get("k_mean", 1.0) for h in hist]
    ok = (mode == "off") and (len(tr) == 0) and all(abs(k - 1.0) < 1e-12 for k in ks)
    return ok, dict(mode=mode, trace_len=len(tr),
                    k_all_one=all(abs(k - 1.0) < 1e-12 for k in ks))


def _max_steps(d):
    c = d.get("config") or {}
    return int(c.get("max_steps", 0)) or None


def _strict_compare(A, R):
    """共有字段逐位比；返回 (bool, info)。缺字段不参与，不假装通过。"""
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
    return ok, dict(points=len(cs), fields=fields,
                    **{("max_d" + k): v for k, v in mx.items()})


def sentinel(A):
    """U3 修正版：同配置未补丁基线优先；max_steps 不同的归档只 SKIP 不判失败。"""
    ms = _max_steps(A)
    # ① 同配置未补丁基线
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
        ok, info = _strict_compare(A, r)
        return ok, dict(ref=name, kind="同配置未补丁基线", max_steps=ms, **info)
    # ② 回退归档；max_steps 不同 ⇒ SKIP
    p = os.path.join(V4, "st25_s0_B.json")
    if not os.path.exists(p):
        return None, dict(reason="缺同配置基线，也缺 st25_s0_B.json")
    r = json.load(open(p, encoding="utf-8"))
    rms = _max_steps(r)
    if ms and rms and ms != rms:
        return None, dict(
            reason="参考系不可比：本档 max_steps=%s，归档 st25_s0_B max_steps=%s"
                   "（warm=max_steps//10 ⇒ lr 调度不同）；"
                   "真 301 点硬前置留给 FULL 关 --zero-check" % (ms, rms),
            ref="st25_s0_B", kind="归档（不同配置）")
    ok, info = _strict_compare(A, r)
    return ok, dict(ref="st25_s0_B", kind="归档（同配置）", **info)


def self_test():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "paperplane_moe"))
    try:
        from enzyme_rate import verify_enzyme_rate
    except ImportError:
        from paperplane_moe.enzyme_rate import verify_enzyme_rate
    good, notes = verify_enzyme_rate(0)
    for n in notes:
        print("  " + n)
    return good


def main():
    if "--self-test" in sys.argv:
        ok = self_test()
        print("FORMULA_SELFTEST=%s" % ("PASS" if ok else "FAIL"))
        raise SystemExit(0 if ok else 1)

    tags = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not tags:
        tags = os.environ.get("SMOKE_TAGS", "").split()
    if not tags:
        raise SystemExit("用法: python3 step29_smoke.py TAG [TAG ...]  或 --self-test")

    print("=" * 78)
    print("step29b 第一关 smoke：三条单元测试 + 同配置零污染哨兵")
    print("V4DIR =", V4, " tags =", " ".join(tags))
    print("=" * 78)

    results = {}
    for t in tags:
        d = L(t)
        mode = d.get("enz_rate_mode", "off")
        print("\n--- %s (mode=%s, max_steps=%s, trace=%d) ---"
              % (t, mode, _max_steps(d), len(d.get("enz_rate_trace", []))))
        if mode == "off":
            ok, info = off_check(d, t)
            print("  [U2] off 档结构零污染: %s  %s" % ("PASS" if ok else "FAIL", info))
            s, si = sentinel(d)
            tag = "PASS" if s is True else ("SKIP" if s is None else "FAIL")
            print("  [U3] 零污染哨兵: %s  %s" % (tag, si))
            results[t] = dict(off_ok=ok, sentinel=s)
        elif mode == "amp" or "freeze" in t:
            ok = (len(d.get("enz_rate_trace", [])) == 0)
            print("  [身份] %s 档 trace 应为空: %s" % (mode, "PASS" if ok else "FAIL"))
            results[t] = dict(identity_ok=ok)
        else:
            T1, T2, T3, info = unit_checks(d, t)
            print("  [T1] f 动态 std=%.4f median=%.4f 范围=[%s,%s]  → %s"
                  % (info["f_std"], info["f_med"], info["f_min"], info["f_max"],
                     "PASS" if T1 else "FAIL"))
            print("  [T2] e 变化 Δ=%.4f                          → %s"
                  % (info["e_delta"], "PASS" if T2 else "FAIL"))
            t3txt = ("PASS" if T3 else "FAIL") if mode in ("cat", "cat_het") \
                else "不适用（k 恒定=设计使然）"
            print("  [T3] corr(k,1/H) mean=%.4f recalc=%.4f      → %s"
                  % (info["corr_mean"], info["corr_recalc"], t3txt))
            print("       k末 mean=%.4f std=%.4f  H=[%s,%s]"
                  % (info["k_last_mean"], info["k_last_std"],
                     info["H_min"], info["H_max"]))
            need = ("T1", "T2", "T3") if mode in ("cat", "cat_het") else ("T1", "T2")
            print("  [门槛] 本档需过 %s（const/lrx 的 k 恒定不算失败）" % (need,))
            results[t] = dict(T1=T1, T2=T2, T3=T3, need=need, info=info)

    UNIT_MODES = ("cat", "cat_het", "const", "lrx")
    cats = [t for t in tags
            if L(t).get("enz_rate_mode") in UNIT_MODES and "freeze" not in t]
    offs = [t for t in tags if L(t).get("enz_rate_mode") == "off"]
    idents = [t for t in tags if t not in cats and t not in offs]
    units_ok = all(all(results[t][k] for k in results[t]["need"])
                   for t in cats) if cats else False
    off_ok = all(results[t]["off_ok"] for t in offs) if offs else True
    sent_ok = all(results[t]["sentinel"] is not False for t in offs) if offs else True
    ident_ok = all(results[t].get("identity_ok", False) for t in idents)
    smoke_ok = units_ok and off_ok and sent_ok and ident_ok and bool(cats) and bool(offs)

    print("\n" + "=" * 78)
    print("[汇总] cats=%s  offs=%s" % (cats, offs))
    print("  T1/T2/T3 全过=%s  off 结构零污染=%s  哨兵=%s  身份=%s"
          % (units_ok, off_ok, sent_ok, ident_ok))
    print("  ★ U3 若 SKIP：真 301 点硬前置仍在 FULL 关 --zero-check，不跳过")
    print("SMOKE_OK=%d   （0 → 别起 300 步 / 3000 步，贴本日志给我）" % int(smoke_ok))
    print("=" * 78)
    raise SystemExit(0 if smoke_ok else 1)


if __name__ == "__main__":
    main()
