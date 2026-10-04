# -*- coding: utf-8 -*-
"""2.4 控制器行为可视化：learned 模式训练 30 epoch，提取每段 lbda/cap/edu 曲线。

输出：
  output/council_behavior_mnist.png   三条曲线（x=训练步/段）
  output/council_log_mnist.csv        完整日志（供报告引用）
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_experiments import make_datasets  # noqa: E402
from marvis_moe_v7 import train_v7  # noqa: E402

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output"))
PNG = os.path.join(OUT_DIR, "council_behavior_mnist.png")
CSV = os.path.join(OUT_DIR, "council_log_mnist.csv")


def main():
    ds = make_datasets("mnist", (2026,), 8000, 2000, pca=64)
    cfg, Xtr, ytr, Xva, yva, Xte, yte = ds[2026]
    model, h, council = train_v7(cfg, Xtr, ytr, Xva, yva, council_mode="learned",
                                 seed=2026, epochs=30, batch_size=256, council_every=20)

    log = h["council_log"]
    print("council segments:", len(log), "val_acc:", round(h["val_acc"][-1], 4),
          "cv:", round(h["final_cv"], 4))

    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "lbda", "cap", "k", "edu"])
        for e in log:
            w.writerow([e["step"], round(e["lbda"], 6), round(e["cap"], 6),
                        e["k"], round(e["edu"], 6)])

    steps = [e["step"] for e in log]
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(steps, [e["lbda"] for e in log], "o-", lw=1.2, ms=3)
    ax[0].set_ylabel("lbda")
    ax[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    ax[1].plot(steps, [e["cap"] for e in log], "s-", lw=1.2, ms=3)
    ax[1].set_ylabel("cap")
    ax[1].axhline(2.0, color="gray", ls="--", lw=0.8)
    ax[2].plot(steps, [e["edu"] for e in log], "^-", lw=1.2, ms=3)
    ax[2].set_ylabel("edu")
    ax[2].set_xlabel("train step (segment end)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("LearnedCouncil behavior on MNIST (PCA64, seed 2026, 30ep)")
    fig.tight_layout()
    fig.savefig(PNG, dpi=130)
    print("saved:", PNG)


if __name__ == "__main__":
    main()
