# -*- coding: utf-8 -*-
"""paperplane-Moe V1.001 · 判据（只读 json，不改源码、不重跑训练）。

用法：python3 step27d_bio_summary.py 2>&1 | tee step27_summary.log

四组数：
  [1] 零污染：bio_s0_B 与 b5_E1_split 的 301 点 loss/eff 与 val
  [2] 主判据：bio_s0_residual − bio_s0_B 的 Δval / 后段逐点 d_loss
  [3] 观测层：events 类别齐全、dissolved_at 必填、谱系可重放
  [4] 参数量对齐 + 红线指标（eff / var_norm / composed_width / 全权重非零）
"""
import json
import os
import statistics as st
import sys

V4 = os.environ.get("V4DIR", "output/v4")
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
try:
    from paperplane_moe.telemetry import validate_events, numeric_projection
except ImportError:
    validate_events = None
    numeric_projection = None


def L(tag):
    p = os.path.join(V4, tag + ".json")
    if not os.path.exists(p):
        raise SystemExit("缺 %s → 训练还没跑完" % p)
    return json.load(open(p))


def paired(a, b, lo=500):
    ha = {q["step"]: q for q in a["history"]}
    hb = {q["step"]: q for q in b["history"]}
    cs = sorted(set(ha) & set(hb))
    ds = [hb[s]["loss"] - ha[s]["loss"] for s in cs if s >= lo]
    assert ds, "后段为空，检查 history 步长"
    return cs, ds


def main():
    try:
        hist, B, R = L("b5_E1_split"), L("bio_s0_B"), L("bio_s0_residual")
    except SystemExit as e:
        print("  ", e)
        return 0

    print("===== [1] 零污染自检（观测层/生物学旁路不得改动 B 档）=====")
    h0 = {q["step"]: q for q in hist["history"]}
    hb = {q["step"]: q for q in B["history"]}
    cs0 = sorted(set(h0) & set(hb))
    mxl = max(abs(h0[s]["loss"] - hb[s]["loss"]) for s in cs0)
    mxe = max(abs((h0[s].get("eff") or 0) - (hb[s].get("eff") or 0)) for s in cs0)
    dv0 = B["val_loss"] - hist["val_loss"]
    print("  对齐点=%d  loss max|Δ|=%.3e  eff max|Δ|=%.3e  Δval=%+.3e" %
          (len(cs0), mxl, mxe, dv0))
    zero_ok = (mxl == 0.0 and mxe == 0.0 and dv0 == 0.0)
    print("  %s" % ("PASS → 零扰动起步成立" if zero_ok else
                    "FAIL → 有混淆变量，先查 RNG / 参数 / clip，别进 A/B"))

    print("\n===== [2] 主判据：残差版真拼 − B（单 seed 效应量）=====")
    cs, ds = paired(B, R)
    dval = R["val_loss"] - B["val_loss"]
    print("  Δval = %+.4f   后段逐点 d_loss = %+.5f ± %.5f  n=%d" %
          (dval, st.mean(ds), st.pstdev(ds), len(ds)))
    ha = {q["step"]: q for q in B["history"]}
    hb = {q["step"]: q for q in R["history"]}
    csteps = sorted(set(ha) & set(hb))
    segs = []
    for lo, hi in [(0, 500), (500, 1500), (1500, 3000)]:
        seg = [hb[s]["loss"] - ha[s]["loss"] for s in csteps if lo <= s < hi]
        if seg:
            segs.append("%d~%d:%+.5f(n=%d)" % (lo, hi, st.mean(seg), len(seg)))
    print("  中途分段: " + "  ".join(segs))
    print("  判读：Δval<0 且后段 d_loss<0 → 残差版赢；"
          "Δval≈0 → 无损但未改善；Δval>0 → 如实关掉")

    print("\n===== [3] 观测层（规格 11/13）=====")
    if validate_events is None:
        print("  SKIP：paperplane_moe 未 import（不影响前两组）")
    else:
        for tag, rec in (("bio_s0_B", B), ("bio_s0_residual", R)):
            v = validate_events(rec.get("events", []), strict=False,
                                observation_only=True)
            print("  %-18s events=%d kinds=%d pass=%s lineage=%s" %
                  (tag, v["n_events"], len(v["kinds_present"]),
                   v["pass"], v["lineage"]))
        print("  numeric_projection 样例（忽略 cell_type 元数据）: %s" %
              (numeric_projection(B["events"][0]) if B.get("events") else {}))

    print("\n===== [4] 参数量 / 红线指标 =====")
    print("  B       params %s→%s  eff=%s  var_norm=%s" %
          (format(B["params_start"], ","), format(B["params_end"], ","),
           B.get("eff_organs_last"), B.get("var_norm_last")))
    print("  res     params %s→%s  eff=%s  var_norm=%s  width=%s" %
          (format(R["params_start"], ","), format(R["params_end"], ","),
           R.get("eff_organs_last"), R.get("var_norm_last"),
           R.get("composed_width")))
    dp = R["params_start"] - B["params_start"]
    print("  Δparams=%+d  (fixed 模式期望 +16 = L×n_cells；"
          "赢归不到容量)" % dp)
    print("  bio_param_overhead(json)=%s" % R.get("bio_param_overhead"))
    print("  disabled_primitives=%s" % R.get("disabled_primitives"))

    print("\n【口径】单 seed 只给效应量；显著需 >=3 seed 配对 + Holm；"
          "不写「加速」（到达步数另需插值验）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
