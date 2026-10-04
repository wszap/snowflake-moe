# -*- coding: utf-8 -*-
"""阶段三：汇总大表 + 显著性检验 + 关键问题回答。

输入：output/results_mnist.csv（本任务），output/experiments_v7.csv 中 synth 行（归档）
输出：output/summary.md
"""
import csv
import os

import numpy as np
from scipy import stats

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output"))
MNIST_CSV = os.path.join(OUT_DIR, "results_mnist.csv")
V7_CSV = os.path.join(OUT_DIR, "experiments_v7.csv")
SUMMARY = os.path.join(OUT_DIR, "summary.md")


def read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def ms(vals):
    a = np.asarray(vals, dtype=float)
    return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0


def wilcoxon_p(x, y):
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if np.all(d == 0):
        return 1.0
    try:
        return float(stats.wilcoxon(x, y).pvalue)
    except ValueError:
        return float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue)


def main():
    mnist = read_rows(MNIST_CSV)
    v7 = [r for r in read_rows(V7_CSV) if r["data"] == "synth"]
    out = []
    w = out.append

    w("# Marvis MoE v7 汇总表（阶段三）\n")
    w("## 3.1 汇总大表（均值 ± 标准差）\n")
    w("| 数据集 | 模式 | val_acc | CV | avg_k | active_frac |")
    w("|---|---|---|---|---|---|")

    def pick(rows, mode):
        return [r for r in rows if r["mode"] == mode]

    for data_name, rows in (("synth", v7), ("mnist", mnist)):
        for mode in ("fixed", "rules", "learned", "learned_nok"):
            rr = pick(rows, mode)
            if not rr:
                continue
            va_m, va_s = ms([r["val_acc"] for r in rr])
            cv_m, cv_s = ms([r["final_cv"] for r in rr])
            ak = [r.get("avg_k", "6.0") or "6.0" for r in rr]
            ak_m, ak_s = ms(ak)
            ac_m, ac_s = ms([r["active"] for r in rr])
            w(f"| {data_name} | {mode} | {va_m:.4f}±{va_s:.4f} | {cv_m:.4f}±{cv_s:.4f} | "
              f"{ak_m:.3f}±{ak_s:.3f} | {ac_m:.2%}±{ac_s:.2%} |")
    w("")

    # ---- 3.2 显著性检验（MNIST，同 seed 配对）----
    w("## 3.2 显著性检验（MNIST，配对样本，n=3 seed）\n")
    w("| 对比 | val_acc 配对差均值 | Wilcoxon p | 配对 t p | 结论 |")
    w("|---|---|---|---|---|")

    def by_mode_seed(rows, mode):
        return {r["seed"]: float(r["val_acc"]) for r in rows if r["mode"] == mode}

    learned = by_mode_seed(mnist, "learned")
    learned_nok = by_mode_seed(mnist, "learned_nok")
    rules = by_mode_seed(mnist, "rules")
    fixed = by_mode_seed(mnist, "fixed")
    seeds = sorted(learned.keys())

    pairs = [
        ("learned vs rules", learned, rules),
        ("learned vs fixed", learned, fixed),
        ("learned_nok vs rules", learned_nok, rules),
    ]
    for name, a, b in pairs:
        x = [a[s] for s in seeds]
        y = [b[s] for s in seeds]
        diff = np.asarray(x) - np.asarray(y)
        wp = wilcoxon_p(x, y)
        tp = float(stats.ttest_rel(x, y).pvalue)
        concl = "无统计显著差异（p>0.05）" if wp > 0.05 and tp > 0.05 else "存在显著差异"
        w(f"| {name} | {diff.mean():+.4f} | {wp:.4f} | {tp:.4f} | {concl} |")
    w("")

    # ---- 3.3 关键问题 ----
    w("## 3.3 关键问题回答\n")
    cv_l_nok = ms([r["final_cv"] for r in pick(mnist, "learned_nok")])
    cv_fixed = ms([r["final_cv"] for r in pick(mnist, "fixed")])
    w(f"**Q1** learned_nok 的 CV（{cv_l_nok[0]:.4f}±{cv_l_nok[1]:.4f}）"
      f"vs fixed 的 CV（{cv_fixed[0]:.4f}±{cv_fixed[1]:.4f}）："
      f"{'低于' if cv_l_nok[0] < cv_fixed[0] else '不低于'} fixed。\n")
    va_learned = ms([r["val_acc"] for r in pick(mnist, "learned")])
    va_rules = ms([r["val_acc"] for r in pick(mnist, "rules")])
    w(f"**Q2** learned 的 val_acc（{va_learned[0]:.4f}±{va_learned[1]:.4f}）"
      f"vs rules 的 val_acc（{va_rules[0]:.4f}±{va_rules[1]:.4f}）："
      f"{'优于' if va_learned[0] > va_rules[0] else '不优于'} rules。\n")
    avg_k_learned = ms([float(r.get("avg_k", 6.0)) for r in pick(mnist, "learned")])
    w(f"**Q3** learned 的 avg_k（{avg_k_learned[0]:.3f}±{avg_k_learned[1]:.3f}），"
      f"主实验 learned 行 avg_k 出现 7.0 / 5.1667 / 5.8571，"
      f"{'存在动态变化（控制器在调整稀疏度）' if avg_k_learned[0] != 6.0 else '接近恒 6（无明显动态）'}。\n")

    text = "\n".join(out)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("saved:", SUMMARY)


if __name__ == "__main__":
    main()
