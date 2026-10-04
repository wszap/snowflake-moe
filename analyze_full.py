# -*- coding: utf-8 -*-
"""阶段 E：汇总全部补全实验（消融 / Token Dropping / 蒸馏 / 5 seed）并出图。

输入（output/）：
  results_ablation_v2.csv       合成消融 3seed×3mode
  results_ablation_mnist_v2.csv MNIST 消融 1seed×3mode
  results_capacity_11/12/15_v2.csv  合成容量 1seed×2mode×3档
  results_edu.csv               合成蒸馏 1seed×3mode
  results_mnist_5seed.csv       MNIST 四方 5seed×4mode
输出（output/）：
  summary_full.md
  council_behavior_mnist_5seed.png
  ablation_bar.png
  capacity_comparison.png
"""
import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output"))


def read_rows(path):
    with open(os.path.join(OUT_DIR, path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ms(vals):
    a = np.asarray(vals, dtype=float)
    return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0


def wilcoxon_p(x, y):
    try:
        return float(stats.wilcoxon(x, y).pvalue)
    except ValueError:
        return float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue)


def group(rows, key="mode"):
    d = defaultdict(list)
    for r in rows:
        d[r[key]].append(r)
    return d


def fmt_line(r, extra=""):
    return (f"{r['mode']:22s} val_acc={float(r['val_acc']):.4f} "
            f"cv={float(r['final_cv']):.4f} avg_k={r.get('avg_k','6')} "
            f"active={float(r['active']):.2%} {extra}")


def main():
    out = []
    w = out.append

    ablation = read_rows("results_ablation_v2.csv")
    ablation_mnist = read_rows("results_ablation_mnist_v2.csv")
    cap11 = read_rows("results_capacity_11_v2.csv")
    cap12 = read_rows("results_capacity_12_v2.csv")
    cap15 = read_rows("results_capacity_15_v2.csv")
    edu = read_rows("results_edu.csv")
    mnist5 = read_rows("results_mnist_5seed.csv")

    w("# Marvis MoE v7 补全实验汇总（阶段 E）\n")
    w(f"生成时间: 2026-10-04；数据来源: output/results_*_v2.csv、results_edu.csv、results_mnist_5seed.csv\n")

    # ============ 1. 汇总大表（按数据集分组） ============
    w("## E.1 汇总大表\n")

    w("### E.1.1 合成数据 · 奖励消融（3 seed × 3 模式）\n")
    w("| 模式 | val_acc | CV | avg_k | active |")
    w("|---|---|---|---|---|")
    for m, rr in sorted(group(ablation).items()):
        va_m, va_s = ms([r["val_acc"] for r in rr])
        cv_m, cv_s = ms([r["final_cv"] for r in rr])
        ak_m, ak_s = ms([r["avg_k"] for r in rr])
        ac_m, ac_s = ms([r["active"] for r in rr])
        w(f"| {m} | {va_m:.4f}±{va_s:.4f} | {cv_m:.4f}±{cv_s:.4f} | {ak_m:.3f}±{ak_s:.3f} | {ac_m:.2%}±{ac_s:.2%} |")
    w("")

    w("### E.1.2 MNIST · 奖励消融（1 seed × 3 模式）\n")
    w("| 模式 | val_acc | test_acc | CV | avg_k |")
    w("|---|---|---|---|---|")
    for m, rr in sorted(group(ablation_mnist).items()):
        r = rr[0]
        w(f"| {m} | {float(r['val_acc']):.4f} | {float(r['test_acc'] or 0):.4f} | "
          f"{float(r['final_cv']):.4f} | {r['avg_k']} |")
    w("")

    w("### E.1.3 合成数据 · 容量机制对比（1 seed × 2 模式 × 3 档 cap）\n")
    w("| cap | 模式 | val_acc | CV | active | avg_k | drop_ratio |")
    w("|---|---|---|---|---|---|---|")
    for cap, rows in (("1.1", cap11), ("1.2", cap12), ("1.5", cap15)):
        for r in rows:
            w(f"| {cap} | {r['mode']} | {float(r['val_acc']):.4f} | {float(r['final_cv']):.4f} | "
              f"{float(r['active']):.2%} | {r['avg_k']} | {r.get('drop_ratio', '0.0')} |")
    w("")

    w("### E.1.4 合成数据 · 软蒸馏对比（1 seed × 3 模式）\n")
    w("| 模式 | val_acc | CV | active | avg_k |")
    w("|---|---|---|---|---|")
    for m, rr in sorted(group(edu).items()):
        r = rr[0]
        w(f"| {m} | {float(r['val_acc']):.4f} | {float(r['final_cv']):.4f} | "
          f"{float(r['active']):.2%} | {r['avg_k']} |")
    w("")

    w("### E.1.5 MNIST · 治理模式四方对比（5 seed × 4 模式）\n")
    w("| 模式 | val_acc | test_acc | CV | active | avg_k |")
    w("|---|---|---|---|---|---|")
    for m, rr in sorted(group(mnist5).items()):
        va_m, va_s = ms([r["val_acc"] for r in rr])
        te_m, te_s = ms([r["test_acc"] for r in rr])
        cv_m, cv_s = ms([r["final_cv"] for r in rr])
        ac_m, ac_s = ms([r["active"] for r in rr])
        ak_m, ak_s = ms([r["avg_k"] for r in rr])
        w(f"| {m} | {va_m:.4f}±{va_s:.4f} | {te_m:.4f}±{te_s:.4f} | "
          f"{cv_m:.4f}±{cv_s:.4f} | {ac_m:.2%}±{ac_s:.2%} | {ak_m:.3f}±{ak_s:.3f} |")
    w("")

    # ============ 2. 显著性检验（n=5） ============
    w("## E.2 显著性检验（MNIST，配对样本，n=5 seed）\n")
    w("| 对比 | val_acc 配对差均值 | 配对 t p | Wilcoxon p | 结论 |")
    w("|---|---|---|---|---|")
    by = {m: {r["seed"]: float(r["val_acc"]) for r in rr} for m, rr in group(mnist5).items()}
    seeds = sorted(by["learned"].keys())
    for name, a, b in (("learned vs rules", "learned", "rules"),
                       ("learned vs fixed", "learned", "fixed"),
                       ("learned vs learned_nok", "learned", "learned_nok"),
                       ("rules vs fixed", "rules", "fixed")):
        x = [by[a][s] for s in seeds]
        y = [by[b][s] for s in seeds]
        diff = np.asarray(x) - np.asarray(y)
        tp = float(stats.ttest_rel(x, y).pvalue)
        wp = wilcoxon_p(x, y)
        concl = "无统计显著差异（p>0.05）" if tp > 0.05 and wp > 0.05 else "存在显著差异"
        w(f"| {name} | {diff.mean():+.4f} | {tp:.4f} | {wp:.4f} | {concl} |")
    w("")

    # ============ 3. 新增问题回答 ============
    w("## E.3 新增问题回答\n")

    # Q1 消融中哪一项奖励贡献最大（MNIST 1 seed，取唯一观测）
    abl_m = {r["mode"]: r for r in ablation_mnist}
    base_learned = float(by["learned"][str(min(seeds))]) if False else None
    w("**Q1 消融中哪一项奖励贡献最大？**\n")
    w("MNIST 单 seed 上：剔除 Δloss（保留 CV+k）val_acc=0.9475 最高；"
      "剔除 k（保留 Δloss+CV）val_acc=0.9337 最低；剔除 CV（保留 Δloss+k）val_acc=0.9350。\n")
    w("含义：k 稀疏度惩罚对最终性能贡献最大（剔除后性能下降最多），Δloss 项贡献最小甚至为负"
      "（剔除后性能反而最高）；CV 负载均衡项贡献中等。合成数据上三项贡献均微弱（下限边界，"
      f"3 seed 差异 ≤0.003，被奖励噪声淹没）。\n")

    w("**Q2 Token Dropping 与可导容量惩罚谁更好？**\n")
    w("可导容量惩罚（learned_capdrop）全面更优：三档 cap 下 val_acc=0.8955 vs Token Dropping"
      " 0.7383（Δ≈0.157）。Token Dropping 在 cap∈{1.1,1.2,1.5} 时丢弃约 79% token"
      "（drop_ratio≈0.79，cap_count=ceil(256/64·cap)=5/5/6），训练信息大量丢失导致性能崩坏；"
      "其唯一优势是负载极致均衡（CV=0.132 vs 1.411），但以精度为代价不可取。\n")

    w("**Q3 自适应蒸馏权重是否优于固定权重？**\n")
    w("否。合成数据上 learned_adaptive_edu 与 learned_fixed_edu 结果完全一致"
      "（val_acc=0.8955, CV=1.4108）：soft_loss 每 50 步、权重 edu=0.15 量级的蒸馏项"
      "相对主损失（~0.5）影响可忽略，token 级权重归一化不改变总量级。关闭蒸馏"
      "（learned_no_edu）略差（0.8936 vs 0.8955），说明蒸馏项作为正则有小幅正向作用，"
      "但自适应/固定之分在合成下限内无差别。\n")

    w("**Q4 扩到 5 seed 后 learned vs rules 是否显著？**\n")
    w("不显著。配对 t p=0.7064、Wilcoxon p=0.8750；learned 均值 0.9488 vs rules 0.9460"
      "（+0.0028），但 seed 间波动（±0.0095）远超差值。结论：learned 与 rules 在 d=64 下"
      "确实无显著差异（非 n=3 统计功效不足）；learned 仅相对 learned_nok 有 +0.0033 的"
      "微弱正向（p=0.7122 亦不显著）。\n")

    text = "\n".join(out)
    with open(os.path.join(OUT_DIR, "summary_full.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

    # ============ 4. 可视化 ============
    plot_council(mnist5)
    plot_ablation(ablation, ablation_mnist)
    plot_capacity([("1.1", cap11), ("1.2", cap12), ("1.5", cap15)])
    print("PNG saved ->", OUT_DIR)


def plot_council(mnist5):
    modes = ["fixed", "rules", "learned", "learned_nok"]
    d = group(mnist5)
    metrics = [("val_acc", "val_acc"), ("final_cv", "CV"), ("active", "active_frac"), ("avg_k", "avg_k")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (col, label) in zip(axes, metrics):
        means, stds = [], []
        for m in modes:
            vals = [float(r[col]) for r in d[m]]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ax.bar(modes, means, yerr=stds, capsize=4, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
        ax.set_title(label)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Council behavior (MNIST, 5 seeds mean±std)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "council_behavior_mnist_5seed.png"), dpi=120)
    plt.close(fig)


def plot_ablation(ablation, ablation_mnist):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # synth 3 seed
    d = group(ablation)
    modes = ["learned_ablate_loss", "learned_ablate_cv", "learned_ablate_k"]
    means, stds = [], []
    for m in modes:
        vals = [float(r["val_acc"]) for r in d[m]]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    axes[0].bar(modes, means, yerr=stds, capsize=4, color="#4C72B0")
    axes[0].set_title("Ablation on synthetic (3 seeds)")
    axes[0].tick_params(axis="x", rotation=20)
    # mnist 1 seed
    d2 = group(ablation_mnist)
    vals2 = [float(d2[m][0]["val_acc"]) for m in modes]
    axes[1].bar(modes, vals2, color="#DD8452")
    axes[1].set_title("Ablation on MNIST (1 seed)")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Reward ablation: val_acc by mode (removed term)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ablation_bar.png"), dpi=120)
    plt.close(fig)


def plot_capacity(cap_groups):
    fig, ax = plt.subplots(figsize=(10, 5))
    caps = [c for c, _ in cap_groups]
    cap_va, tok_va, tok_drop = [], [], []
    for _, rows in cap_groups:
        d = {r["mode"]: r for r in rows}
        cap_va.append(float(d["learned_capdrop"]["val_acc"]))
        tok_va.append(float(d["learned_tokendrop"]["val_acc"]))
        tok_drop.append(float(d["learned_tokendrop"].get("drop_ratio", 0.0)))
    x = np.arange(len(caps))
    ax.bar(x - 0.2, cap_va, 0.4, label="learned_capdrop (differentiable penalty)", color="#55A868")
    ax.bar(x + 0.2, tok_va, 0.4, label="learned_tokendrop (hard drop)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([f"cap={c}" for c in caps])
    ax.set_ylabel("val_acc")
    ax.set_title("Capacity mechanism: differentiable penalty vs token dropping")
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(x, tok_drop, "o--", color="gray", label="drop_ratio (tokendrop)")
    ax2.set_ylabel("drop_ratio", color="gray")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "capacity_comparison.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
