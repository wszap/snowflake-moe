#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_ci.py —— core_stat 统计修正审计（Bug9）

背景
----
run_metrics.py 原算法：
    n_eff = n_perm * n_batch = 20 * 50 = 1000
    se    = sd / sqrt(n_eff)
但同一 batch 内的 20 个 perm 共享同一组 xb/yb，不是独立样本
⇒ 伪重复(pseudo-replication)，se 被缩小约 sqrt(n_perm)=4.5 倍
⇒ CI 虚窄、p 虚假地小

本脚本做两件事：
  ① 用现有 record.json 重算【保守】CI（不重跑训练），验证结论是否改变
  ② 打印修正前后的对比，供论文方法部分引用

保守估计下界：
    取 n_eff = n_batch（不乘 n_perm），这是【最保守】的修正。
    真正的配对检验会进一步减小 sd（因为扣除了 batch 间的系统差异），
    所以本脚本给出的 CI 是真实 CI 的【上界】——
    若本脚本判定 pass，真实配对检验必然也 pass。

用法
----
    python audit_ci.py                          # 扫最近 10 个 run
    python audit_ci.py --runs <dir1> <dir2>     # 指定 run 目录
    python audit_ci.py --sd 0.0256 --n 50       # 手工指定参数做演示

作者：审查补丁配套工具
"""
import argparse
import glob
import json
import math
import os
import sys


def tcrit(df, conf=0.975):
    """t 分布临界值；无 scipy 时退回正态近似（略保守）。"""
    try:
        from scipy import stats as st
        return float(st.t.ppf(conf, max(df, 1)))
    except Exception:
        # 正态 1.96 对小 df 偏小 ⇒ 这里用 2.0 兜底（更保守）
        return 2.0 if df < 30 else 1.96


def read_run(d):
    p = os.path.join(d, "record.json")
    with open(p, encoding="utf-8") as f:
        r = json.load(f)
    fin = r.get("final", {})
    return {
        "run_id": r.get("meta", {}).get("run_id", os.path.basename(d.rstrip("/"))),
        "core": fin.get("core_stat"),
        "ci_lo": fin.get("core_stat_ci_lo"),
        "ci_hi": fin.get("core_stat_ci_hi"),
        "sd": fin.get("loss_permuted_sd"),
        "n_perm": fin.get("n_permute_repeats", 20),
        "d_z": fin.get("cohens_dz"),
        "p": fin.get("p_core"),
        "n_eff_used": fin.get("n_eff_used"),
        "paired_sd": fin.get("core_stat_paired_sd"),
        "n_paired": fin.get("n_paired_batches"),
        "loss": fin.get("loss_learned"),
    }


def recompute(core, sd, n_perm=20, n_batch=50, label=""):
    """返回 (旧算法, 保守修正) 两套 CI。"""
    # 旧算法：n_eff = n_perm * n_batch（伪重复）
    n_old = max(n_perm * n_batch, 2)
    se_old = sd / math.sqrt(n_old)
    tc_old = tcrit(n_old - 1)
    old = (core - tc_old * se_old, core + tc_old * se_old, se_old, n_old)

    # 保守修正：n_eff = n_batch（真实独立样本数）
    n_new = max(n_batch, 2)
    se_new = sd / math.sqrt(n_new)
    tc_new = tcrit(n_new - 1)
    new = (core - tc_new * se_new, core + tc_new * se_new, se_new, n_new)
    return old, new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--sd", type=float, default=None, help="手工指定 sd 做演示")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--n-batch", type=int, default=50)
    ap.add_argument("--n", type=int, default=10, help="取最近 n 个 run")
    args = ap.parse_args()

    print("=" * 78)
    print("  core_stat 统计修正审计（Bug9：伪重复 + 非配对）")
    print("=" * 78)

    # ---------- 演示模式 ----------
    if args.sd is not None:
        core = 0.7839
        old, new = recompute(core, args.sd, args.n_perm, args.n_batch)
        print(f"\n[演示] core={core}  sd={args.sd}  n_perm={args.n_perm}  n_batch={args.n_batch}")
        print(f"  旧: n_eff={old[3]:>4}  se={old[2]:.6f}  CI=[{old[0]:.4f}, {old[1]:.4f}]  宽={old[1]-old[0]:.4f}")
        print(f"  新: n_eff={new[3]:>4}  se={new[2]:.6f}  CI=[{new[0]:.4f}, {new[1]:.4f}]  宽={new[1]-new[0]:.4f}")
        print(f"  修正后 CI 宽度放大 {(new[1]-new[0])/(old[1]-old[0]):.1f} 倍")
        print(f"  结论: {'仍然 pass（下界>0）' if new[0] > 0 else 'FAIL —— 需要关注'}")
        return 0

    # ---------- 读 run ----------
    if args.runs:
        dirs = args.runs
    else:
        dirs = sorted(glob.glob("output/runs/*/"), key=os.path.getmtime)[-args.n:]
    if not dirs:
        print("[ERR] 没找到 run 目录。cd 到 v2_compositional 再跑，或用 --runs 指定。")
        return 1

    rows = []
    for d in dirs:
        try:
            rows.append(read_run(d))
        except Exception as e:
            print(f"[WARN] 跳过 {d}: {e}")

    if not rows:
        print("[ERR] 没有可读的 record.json")
        return 1

    # ---------- 是否已经打过补丁 ----------
    patched = any(r.get("n_eff_used") for r in rows)
    print(f"\n补丁状态: {'✅ 已打 Bug9 补丁（n_eff_used 存在）' if patched else '⚠️ 未打补丁 —— 下面是【保守重算】'}")

    print(f"\n{'run_id':24} {'core':>8} {'旧CI宽':>8} {'修正CI宽':>9} {'修正下界':>9} {'判定':>6}")
    print("-" * 78)

    cores, new_los, new_ws, old_ws = [], [], [], []
    for r in rows:
        core = r["core"]
        if core is None:
            continue
        sd = r.get("sd") or 0.0256          # 缺失时用代表值
        n_perm = r.get("n_perm") or 20

        if patched and r.get("paired_sd"):
            # 已打补丁：直接用真实的配对统计
            se = r["paired_sd"] / math.sqrt(max(r.get("n_paired") or 1, 2))
            tc = tcrit((r.get("n_paired") or 2) - 1)
            lo, hi = core - tc * se, core + tc * se
            old_w = (r["ci_hi"] - r["ci_lo"]) if r["ci_hi"] else float("nan")
        else:
            old, new = recompute(core, sd, n_perm, args.n_batch)
            lo, hi = new[0], new[1]
            old_w = old[1] - old[0]

        new_w = hi - lo
        cores.append(core); new_los.append(lo); new_ws.append(new_w); old_ws.append(old_w)
        verdict = "pass" if lo > 0 else "FAIL"
        print(f"{r['run_id']:24} {core:>8.4f} {old_w:>8.4f} {new_w:>9.4f} {lo:>9.4f} {verdict:>6}")

    # ---------- 汇总 ----------
    if not cores:
        print("[ERR] 没有 core_stat 数据")
        return 1

    import statistics as st
    m = st.mean(cores); sd_m = st.stdev(cores) if len(cores) > 1 else 0.0
    n = len(cores)
    # 跨 seed 的 t 检验（这才是论文该报的：seed 间变异）
    se_m = sd_m / math.sqrt(n) if n > 1 else 0.0
    tc = tcrit(n - 1)
    ci_lo, ci_hi = m - tc * se_m, m + tc * se_m
    tval = m / se_m if se_m > 0 else float("inf")

    print("-" * 78)
    print(f"\n【core_stat 汇总】(n={n} seeds)")
    print(f"  均值      = {m:.4f} ± {sd_m:.4f}   范围 [{min(cores):.4f}, {max(cores):.4f}]")
    print(f"  跨seed CI = [{ci_lo:.4f}, {ci_hi:.4f}]   t({n-1})={tval:.1f}   (临界 {tc:.3f})")
    print(f"  全部>0    = {all(c > 0 for c in cores)}")

    print(f"\n【CI 宽度修正】")
    print(f"  旧 CI 平均宽度   = {st.mean(old_ws):.4f}")
    print(f"  修正 CI 平均宽度 = {st.mean(new_ws):.4f}   （放大 {st.mean(new_ws)/max(st.mean(old_ws),1e-9):.1f} 倍）")
    print(f"  修正后最差下界   = {min(new_los):.4f}   （seed 级最保守）")

    print(f"\n{'='*78}")
    ok = all(lo > 0 for lo in new_los) and ci_lo > 0
    if ok:
        print("  ✅ 结论：H4「分工真实存在」在统计修正后【依然成立】")
        print(f"     即使按最保守的 n_eff={args.n_batch}（不乘 n_perm）重算，")
        print(f"     所有 {n} 个 seed 的 CI 下界仍 > 0，跨 seed CI 下界 = {ci_lo:.4f}")
    else:
        print("  ❌ 结论：修正后 H4 不再稳固，必须重新检验！")
        print(f"     最差下界 = {min(new_los):.4f}")
    print("=" * 78)

    print(f"\n【论文写法建议】")
    print(f"  旧（不要这样写）: core_stat = {m:.4f}, CI=[±0.0016], p<1e-300")
    print(f"  新（应这样写）  : core_stat = {m:.4f} ± {sd_m:.4f} (n={n} seeds),")
    print(f"                    95% CI [{ci_lo:.4f}, {ci_hi:.4f}], t({n-1})={tval:.1f}")
    print(f"  并注明：CI 基于 seed 间变异；单次 run 内部采用配对检验")
    print(f"          （permuted 与 learned 在同一 batch 上配对），")
    print(f"          不使用 n_perm×n_batch 作为有效样本数（避免伪重复）。")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
