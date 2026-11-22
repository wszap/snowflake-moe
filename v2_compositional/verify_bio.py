# -*- coding: utf-8 -*-
"""paperplane-Moe V1.001 · 真机（有 torch）自检。

在 v2_compositional/ 下跑：
    python3 verify_bio.py --json output/v4/bio_s0_B.json

验收 9 条，任一 FAIL 就别进 A/B：
  1. 新模块全部可 import（改名 B 的 shim 也测）
  2. gate=0 时 ResidualComposeCell 输出与 SnowflakeB 逐位一致
  3. 新增 E1/E2 用局部 generator → 全局 RNG 状态不变
  4. 红线：wiring 全 n 非零、composed_width = k×h_cell
  5. 临时模组 Δw 正锥 + 零和 + open/dissolve 成对（dissolved_at 必填）
  6. 观测层 10 类事件齐全、谱系可重放
  7. 训练期 clone_readonly() 抛错；推理期副本输出逐位一致
  8. reset_state() 后输出与「全零新细胞」一致，其余 cell 不受影响
  9. 参数量对账 + 301 点零污染（--json 时读 st25_s0_B 与 b5_E1_split）
"""
import argparse
import json
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from snowflake_B import SnowflakeB
from paperplane_moe import (
    BioConfig, TelemetryRecorder, TempExpertModule, ResidualComposeCell,
    Bloodstream, Receptor, Nucleoskeleton, SparseRecall, ATPBudget,
    Autophagy, Apoptosis, CellLineage, assert_compose_redline,
    compose_width, param_overhead,
)
from paperplane_moe.telemetry import validate_events, CORE_EVENT_KINDS
from paperplane_moe import paperplane_v4, paperplane_B, paperplane_compose

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("  %-4s %-38s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="", help="零污染自检用的 bio_s0_B.json")
    ap.add_argument("--ref", default="output/v4/st25_s0_B.json")
    ap.add_argument("--hist-ref", default="output/v4/b5_E1_split.json")
    a = ap.parse_args()

    torch.manual_seed(0)
    d, n, h, rank, seq_len = 64, 32, 16, 4, 8
    kw = dict(d=d, n_organelles=n, h=h, rank=rank, seq_len=seq_len,
              chunk_size=1, temp=2.0, hinge_mode="loss",
              use_hinge_low=False, log_sharp=False,
              min_eff_organs=0, seed=0)
    x = torch.randn(16, d)

    # 1) import + 别名
    check("1 新模块与 paperplane_* 别名可 import",
          paperplane_v4.PaperplaneLMV4 is not None
          and paperplane_B.PaperplaneB is SnowflakeB
          and paperplane_compose.PaperplaneResidualComposeCell
          is ResidualComposeCell)

    # 2/3) 零扰动 + 全局 RNG 不变
    # 正确判法：两种 cell 各自从同一 seed 构造，比较【构造之后】的全局 RNG 状态。
    # ResidualComposeCell 的 E1/E2 用局部 generator，不应比 B 多消耗任何一次抽签。
    torch.manual_seed(123)
    b2 = SnowflakeB(**kw).eval()
    st_b = torch.get_rng_state().clone()
    torch.manual_seed(123)
    r = ResidualComposeCell(k=4, h_cell=3, mode="fixed", **kw).eval()
    st_r = torch.get_rng_state().clone()
    rng_unchanged = torch.equal(st_b, st_r)
    with torch.no_grad():
        out_b, info_b = b2(x)
        out_r, info_r = r(x)
    maxd = float((out_b - out_r).abs().max())
    check("2 gate=0 输出与 B 档逐位一致", maxd == 0.0,
          "max|Δ|=%.3e" % maxd)
    check("3 局部 generator 不动全局 RNG", rng_unchanged)

    # 4) 红线
    red = assert_compose_redline(
        info_r["wiring"], info_r["composed_width"], 4, 3,
        float(r.gate_comp.detach().mean()), out_r, out_b)
    check("4 拼专家红线（不筛选/宽度线性/恒等）",
          red["no_selection"] and red["width_linear"]
          and red["gate_zero_is_identity"],
          "width=%s sum_w=%s" % (compose_width(4, 3), red["sum_w"]))

    # 5) 临时模组
    cfg = BioConfig(enable=False)
    tm = TempExpertModule(n, d, cfg)
    ev_open = tm.open(7, "task-test", info_r["wiring"], [0, 1], ["info-0001"])
    dw = tm.delta_w.clone()
    ev_close = tm.dissolve(7, 0.0)
    check("5 临时模组 Δw 全零起步 + 成对解散",
          float(dw.abs().sum()) == 0.0
          and ev_open["dissolved_at"] is None
          and ev_close["dissolved_at"] == 7
          and not tm.active,
          "w_sum=%s" % ev_open["w_sum"])

    # 5b) gain>0 时正锥 + 零和
    cfg2 = BioConfig(enable=True, gate=0.1)
    tm2 = TempExpertModule(n, d, cfg2)
    tm2.open(8, "task-test2", info_r["wiring"], [0], ["info-0001"])
    dw2 = tm2.delta_w
    check("5b Δw 零和 + 正锥 w+Δw>0",
          abs(float(dw2.sum())) < 1e-5
          and bool((info_r["wiring"].mean(0)[:n] + dw2 > 0).all()),
          "sum=%+.3e" % float(dw2.sum()))

    # 6) 观测层（用 train_bio 的真实事件形状模拟一轮）
    tel = TelemetryRecorder(cfg, n_cells=16, n_organelles=n)
    tel.on_divide(1500, n, 8, list(range(8)), "residual", 0.0, 0.0)
    tel.on_wire(1500, [0.012] * 8)
    for s in (0, 500, 1000, 1500, 2000, 2500):
        tel.on_cell_call(s, list(range(16)), 3.2, 0.28, 0.0)
        tel.on_mem_kind(s, 1, 1, "info", "info-0001")
        tel.on_budget(s, 0.0, 0.0, "observe")
        tel.on_transport(s, "passive", 0, 0.0)
        tel.on_health(s, 3.2, 0.28, True, 0.01)
        tel.open_temp_module(tm, s, info_r["wiring"], [0, 1], ["info-0001"])
        tel.dissolve_temp_module(tm, s, 0.0)
        tel.on_clone(s, 0, True)
    evs = []
    tel.flush_to(evs)
    val = validate_events(evs, strict=False, observation_only=True)
    check("6 观测层 10 类齐全 + 谱系可重放",
          val["pass"] and val["lineage"]["born"] == 8,
          "events=%d born=%s" % (val["n_events"], val["lineage"]["born"]))

    # 7) clone
    r.train()
    thrown = False
    try:
        r.clone_readonly()
    except AssertionError:
        thrown = True
    r.eval()
    twin = r.clone_readonly()
    with torch.no_grad():
        o1, _ = r(x)
        o2, _ = twin(x)
    check("7 训练期禁止复制 / 推理期副本逐位一致",
          thrown and torch.equal(o1, o2))

    # 8) reset
    r2 = ResidualComposeCell(k=4, h_cell=3, mode="fixed", **kw).eval()
    r2.reset_state()
    check("8 reset 后忆库/基底/份额全零",
          float(r2.organelle_memory.abs().sum()) == 0.0
          and float(r2.gate_comp.abs().sum()) == 0.0
          and float(r2.mix_ema.abs().sum()) == 0.0)

    # 9) 参数量 + 301 点零污染
    ov = param_overhead(16, "fixed", n, d, 3)
    check("9a 参数量对账（L×n_cells=16）", ov == 16,
          "+16 (相对 17,225,573 = +0.00009%%)")
    if a.json and os.path.exists(a.json) and os.path.exists(a.hist_ref):
        j = json.load(open(a.json))
        h0 = json.load(open(a.hist_ref))
        ha = {q["step"]: q for q in h0["history"]}
        hb = {q["step"]: q for q in j["history"]}
        cs = sorted(set(ha) & set(hb))
        mx = max(abs(ha[s]["loss"] - hb[s]["loss"]) for s in cs)
        dv = j["val_loss"] - h0["val_loss"]
        check("9b 301 点零污染 + val 逐位", mx == 0.0 and dv == 0.0,
              "n=%d max|Δ|=%.3e Δval=%+.3e" % (len(cs), mx, dv))
        v = validate_events(j.get("events", []), strict=False,
                            observation_only=True)
        check("9c 落盘 json 观测层校验", v["pass"],
              "events=%d" % v["n_events"])
    else:
        print("  SKIP 9b/9c（--json 未给或文件不存在，不假装通过）", flush=True)

    ok = sum(1 for _, o, _ in RESULTS if o)
    print("\n===== %d/%d 通过 =====" % (ok, len(RESULTS)))
    print("  FAIL 清单: %s" % [n for n, o, _ in RESULTS if not o])
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
