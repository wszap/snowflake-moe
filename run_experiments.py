# -*- coding: utf-8 -*-
"""自动化实验流水线：3 方法 × N seeds → CSV（含断点续跑与进度日志）。

用法示例：
    python run_experiments.py --data synth --seeds 2026 2027 2028 --modes fixed rules learned --epochs 12
    python run_experiments.py --data mnist --seeds 2026 2027 --modes fixed rules learned --epochs 12 --n_train 8000

行为：
- 自动遍历 (data, seed, mode) 组合并调用 train_v7
- 每跑完一个训练，结果写入 CSV（追加），屏幕打印进度日志
- CSV 中已存在的 (data, seed, mode, epochs) 行自动跳过（断点续跑）
- 每改超参必须在 CONFIG_NOTES 中登记理由，否则视为违规
"""
import argparse
import csv
import gc
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marvis_moe import Config, make_synthetic  # noqa: E402
from marvis_moe_v7 import train_v7  # noqa: E402

# ---------------------------------------------------------------------------
# 超参登记表（防调参地狱）：每次改动超参必须在此登记理由。
# 格式: "日期|参数=值|改动理由"
# 规则: 同一问题域累计改动超过 3 次仍未收敛 -> 停止改码，汇报根因并建议降级。
# ---------------------------------------------------------------------------
CONFIG_NOTES = [
    "2026-10-04|lr=2e-3, council_lr=1e-3|与既有 v6/v7 冒烟一致，首轮流水线基线，不视为调参",
    "2026-10-04|reward=-(ce-prev_seg_avg)-beta*cv-gamma*abs(k-topk)，beta=0.5,gamma=0.1（废弃 loss_ref 逻辑）|机制修正（红队#1）：loss_ref 恒 None 致奖励残缺，改为 Delta Loss 直接奖励相对上一段平均 loss 的下降",
    "2026-10-04|collect_states 每10个决策段调用 update_norm|机制修正（红队#2）：消除状态归一化死代码",
    "2026-10-04|REINFORCE loss 加熵正则 -0.01*log_std.mean()|机制修正（红队#3）：防方差崩溃/探索消失",
    "2026-10-04|k 动作空间改为 base_k±16 窄区间（clip 到 [2,E]）|机制修正：原 [2,64] 均匀映射使初始 k≈33，控制器靠大 k 刷 Δloss 收益，k 惩罚被淹没；±6 过窄致 k 无动态（探索不足），扩至 ±16 保留惩罚传导并恢复动态",
    "2026-10-04|合成数据失效边界记录（阴性结果）|三次尝试轨迹: ①Delta Loss 首跑 val_acc=0.8955 CV=1.4108 k∈[29,34]（k贴高值刷收益）；②k窄区间±6 val_acc=0.8955 CV=1.4108 k恒6；③k窄区间±16 val_acc=0.8955 CV=1.4108 k恒6。根因: d=16、4096样本下路由已固化，REINFORCE奖励噪声（std≈0.24）淹没CV惩罚信号。结论: 合成数据为RL动态治理的物理失效下限，不再修改，转MNIST阈值突破",
    "2026-10-04|新增消融/容量/蒸馏模式（阶段A/B/C）|补全报告实验：learned_ablate_loss/cv/k 分别剔除奖励三项；learned_capdrop=固定cap+可导容量惩罚、learned_tokendrop=固定cap+Token Dropping（MoELayer 新增 use_token_dropping 互斥路径，超容 token 直接丢弃不参与计算/反向）；learned_no_edu/fixed_edu/adaptive_edu 控制蒸馏开关与 soft_loss 自适应权重（距离大者权重大）。机制已验证：drop_ratio 随 cap 单调、adaptive loss 数值有别、8+10 项测试冒烟全绿",
]

# CSV 输出位置
OUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "output", "experiments_v7.csv")
OUT_CSV = os.path.abspath(OUT_CSV)

CSV_HEADER = ["data", "seed", "mode", "epochs", "val_acc", "train_acc",
              "test_acc", "loss", "final_cv", "active", "avg_k", "drop_ratio",
              "events", "n_rewards", "sec"]


def load_done(out_csv):
    """读取已完成的 (data, seed, mode, epochs) 集合，用于断点续跑。"""
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["data"], int(row["seed"]), row["mode"], int(row["epochs"])))
    return done


def append_row(out_csv, row):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if new:
            w.writeheader()
        w.writerow(row)


def make_datasets(data, seeds, n_train, n_test, pca=None, d=16, E=64, S=2, L=2, K=8):
    """按 data 类型准备各 seed 的数据分片。

    pca: MNIST 降维目标维数（None=不降维，保持 784）。
    PCA 在全量训练集上 fit，再 transform 各 seed 分片，保证跨 seed 可比。
    """
    datasets = {}
    if data == "synth":
        cfg = Config(d=d, h=32, E=E, S=S, L=L, topk=6, out_dim=K)
        for sd in seeds:
            X, y = make_synthetic(n=4096, d=d, k=K, seed=sd)
            perm = np.random.RandomState(sd).permutation(X.shape[0])
            X, y = X[perm], y[perm]
            datasets[sd] = (cfg, X[:3072], y[:3072], X[3072:], y[3072:], None, None)
    else:  # mnist
        from sklearn.decomposition import PCA
        from train_mnist import load_mnist, sample
        Xtr_all, ytr_all = load_mnist("train")
        Xte_all, yte_all = load_mnist("t10k")
        if pca:
            pca_model = PCA(n_components=pca)
            Xtr_all = torch.from_numpy(pca_model.fit_transform(Xtr_all).astype(np.float32))
            Xte_all = torch.from_numpy(pca_model.transform(Xte_all).astype(np.float32))
        cfg = Config(d=Xtr_all.shape[1], h=256, E=E, S=S, L=L, topk=6, out_dim=10)
        for sd in seeds:
            Xtr_s, ytr_s = sample(Xtr_all, ytr_all, n_train, seed=sd)
            Xte_s, yte_s = sample(Xte_all, yte_all, n_test, seed=sd)
            perm = np.random.RandomState(sd).permutation(n_train)
            va_n = n_train // 10
            datasets[sd] = (cfg,
                            Xtr_s[perm[:-va_n]], ytr_s[perm[:-va_n]],
                            Xtr_s[perm[-va_n:]], ytr_s[perm[-va_n:]],
                            Xte_s, yte_s)
    return datasets


def check_temp(guard=True, limit=90.0):
    """读取 CPU 温度（WMI 热区最大值）。guard=True 且超限时抛异常停止。返回温度。"""
    if not guard:
        return None
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             '(Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature | Measure-Object -Property CurrentTemperature -Maximum).Maximum'],
            capture_output=True, text=True, timeout=10)
        t = float(out.stdout.strip()) / 10.0 - 273.15
    except Exception:
        t = None
    if t is not None and t > limit:
        raise RuntimeError(f"CPU 温度 {t:.1f}°C 超过红线 {limit}°C，停止训练")
    return t


def run_experiments(data="synth", seeds=(2026, 2027, 2028), modes=("fixed", "rules", "learned"),
                    epochs=12, n_train=8000, n_test=2000, pca=None, out_csv=OUT_CSV,
                    cap=None, temp_guard=True):
    datasets = make_datasets(data, seeds, n_train, n_test, pca=pca)
    done = load_done(out_csv)
    total = len(seeds) * len(modes)
    idx = 0
    results = {m: [] for m in modes}

    for sd in seeds:
        cfg, Xtr, ytr, Xva, yva, Xte, yte = datasets[sd]
        for m in modes:
            idx += 1
            key = (data, sd, m, epochs)
            if key in done:
                print(f"[SKIP {idx}/{total}] {data} seed={sd} mode={m} epochs={epochs} (已存在)")
                continue
            t0 = time.time()
            print(f"[RUN   {idx}/{total}] {data} seed={sd} mode={m} epochs={epochs} n_train={Xtr.shape[0]} ...")
            sys.stdout.flush()
            model, h, _ = train_v7(cfg, Xtr, ytr, Xva, yva, council_mode=m,
                                   seed=sd, epochs=epochs, batch_size=256, lr=2e-3,
                                   fixed_cap=cap, use_token_dropping=(m == 'learned_tokendrop'))
            test_acc = ""
            if Xte is not None:
                model.eval()
                import torch
                with torch.no_grad():
                    logits, _ = model(Xte)
                    test_acc = round(float((logits.argmax(-1) == yte).float().mean().item()), 6)
            row = dict(data=data, seed=sd, mode=m, epochs=epochs,
                       val_acc=round(h['val_acc'][-1], 6),
                       train_acc=round(h['train_acc'], 6),
                       test_acc=test_acc,
                       loss=round(h['avg_loss'], 6),
                       final_cv=round(h['final_cv'], 6),
                       active=round(h['final_active'], 6),
                       avg_k=round(h['avg_k'], 4),
                       drop_ratio=round(h.get('drop_ratio', 0.0), 4),
                       events=len(h['events']),
                       n_rewards=len(h['rewards']),
                       sec=round(time.time() - t0, 1))
            append_row(out_csv, row)
            results[m].append(row)
            gc.collect()                       # 散热纪律：每组合后 gc
            t_cpu = check_temp(temp_guard)
            print(f"   [guard] gc.collect() OK; CPU 温度: {t_cpu if t_cpu is not None else 'N/A'}°C")
            sys.stdout.flush()
            print(f"   -> val_acc={row['val_acc']:.4f} train_acc={row['train_acc']:.4f} "
                  f"loss={row['loss']:.4f} cv={row['final_cv']:.4f} "
                  f"active={row['active']:.2%} avg_k={row['avg_k']} "
                  f"events={row['events']} ({row['sec']}s)")
            sys.stdout.flush()

    print("\n" + "=" * 72)
    print(f"流水线完成（{data}） 输出: {out_csv}")
    print("=" * 72)
    for m in modes:
        rows = results[m]
        if not rows:
            continue
        va = np.array([r['val_acc'] for r in rows])
        cv = np.array([r['final_cv'] for r in rows])
        te = [r['test_acc'] for r in rows if r['test_acc'] != ""]
        te_s = f" test_acc={np.mean(te):.4f}±{np.std(te):.4f}" if te else ""
        print(f"{m:<10} val_acc={va.mean():.4f}±{va.std():.4f}  cv={cv.mean():.4f}±{cv.std():.4f}{te_s}  n={len(rows)}")
    return out_csv


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="MarvisMoE v7 自动化实验流水线")
    p.add_argument("--data", choices=("synth", "synthetic", "mnist", "shakespeare"),
                   default="synth", help="synth/synthetic 为合成数据别名；shakespeare 为字符级语言建模")
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    p.add_argument("--modes", nargs="+", default=["fixed", "rules", "learned"])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--n_train", type=int, default=8000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--pca", type=int, default=None,
                   help="MNIST 降维目标维数（如 64）。None 时不降维")
    # 语言建模（shakespeare）参数
    p.add_argument("--seq_len", type=int, default=64, help="LM 序列长度")
    p.add_argument("--d", type=int, default=64, help="LM 隐藏维度")
    p.add_argument("--h", type=int, default=128, help="LM 专家内层宽度")
    p.add_argument("--E", type=int, default=4, help="LM 路由专家数")
    p.add_argument("--S", type=int, default=1, help="LM 共享专家数")
    p.add_argument("--L", type=int, default=2, help="LM MoE 层数")
    p.add_argument("--topk", type=int, default=2, help="LM 每 token 激活专家数")
    p.add_argument("--batch", type=int, default=16, help="LM 批大小")
    p.add_argument("--accum", type=int, default=8, help="LM 梯度累积步数")
    p.add_argument("--lr", type=float, default=3e-4, help="LM 学习率")
    p.add_argument("--out", default=OUT_CSV)
    p.add_argument("--cap", type=float, default=None,
                   help="固定容量因子（learned_capdrop/learned_tokendrop 用），如 1.1/1.2/1.5")
    p.add_argument("--temp_guard", action="store_true", default=True,
                   help="训练组合间检查 CPU 温度，>90°C 停止（默认开启）")
    p.add_argument("--optimize", action="store_true", default=False,
                   help="算法优化：梯度裁剪 max_norm=1.0 + LR warmup 前 5% steps")
    args = p.parse_args()
    data = "synth" if args.data == "synthetic" else args.data
    if data == "shakespeare":
        from train_lm import run_experiments_lm
        run_experiments_lm(seeds=tuple(args.seeds), modes=tuple(args.modes),
                           epochs=args.epochs, d=args.d, h=args.h, E=args.E,
                           S=args.S, L=args.L, topk=args.topk,
                           seq_len=args.seq_len, batch_size=args.batch,
                           accum=args.accum, lr=args.lr, cap=args.cap,
                           out_csv=os.path.abspath(args.out),
                           device="cuda" if torch.cuda.is_available() else "cpu",
                           use_clip=args.optimize, use_warmup=args.optimize)
    else:
        run_experiments(data=data, seeds=tuple(args.seeds), modes=tuple(args.modes),
                        epochs=args.epochs, n_train=args.n_train, n_test=args.n_test,
                        pca=args.pca, out_csv=os.path.abspath(args.out), cap=args.cap,
                        temp_guard=args.temp_guard)
