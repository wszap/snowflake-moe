# -*- coding: utf-8 -*-
"""Marvis MoE v5 真实训练：MNIST（CPU, torch 2.14.1）
复用 marvis_moe.py 的 MarvisMoE 类，解析原始 MNIST 数据训练，公平对比治理开/关。
"""
import gzip
import struct
import time

import numpy as np
import torch
from sklearn.decomposition import PCA

from marvis_moe import Config, MarvisMoE, set_seed

MNIST_DIR = r"D:\\snowflakeAI"


def load_mnist(kind):
    """kind: 'train' / 't10k'。返回 (X: float32 [N,784] 0~1, y: int64 [N])"""
    if kind == "train":
        img_f, lab_f = "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"
    else:
        img_f, lab_f = "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    with gzip.open(f"{MNIST_DIR}\\{img_f}", "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    with gzip.open(f"{MNIST_DIR}\\{lab_f}", "rb") as f:
        magic, n2 = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8)
    return torch.from_numpy(X.astype(np.float32) / 255.0), torch.from_numpy(y.astype(np.int64))


def sample(X, y, n, seed=2026):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], n, replace=False)
    return X[idx], y[idx]


def apply_pca(Xtr, Xte, n_components=64):
    """PCA 降维：从 784 维降到 n_components（默认 64）。

    在全量训练集上 fit，再 transform 测试集，保证同分布可比。
    输入输出均为 torch tensor。
    """
    pca = PCA(n_components=n_components)
    Xtr_np = Xtr.numpy() if torch.is_tensor(Xtr) else Xtr
    Xte_np = Xte.numpy() if torch.is_tensor(Xte) else Xte
    Xtr_p = pca.fit_transform(Xtr_np).astype(np.float32)
    Xte_p = pca.transform(Xte_np).astype(np.float32)
    return torch.from_numpy(Xtr_p), torch.from_numpy(Xte_p)


def run_gov(gov, cfg, Xtr, ytr, Xte, yte, epochs, batch_size, lr):
    set_seed(2026)
    model = MarvisMoE(cfg)
    t0 = time.time()
    h = model.fit(Xtr, ytr, epochs=epochs, batch_size=batch_size, lr=lr,
                  gov=gov, council_every=25, edu_every=60)
    dt = time.time() - t0
    # 测试集评估
    model.eval()
    with torch.no_grad():
        logits, _ = model(Xte)
        acc = (logits.argmax(-1) == yte).float().mean().item()
    loss = np.mean(h["loss"][-30:])
    return {
        "train_acc": np.mean(h["acc"][-30:]),
        "test_acc": acc,
        "loss": loss,
        "std": h["final_util_std"],
        "active": h["final_active"],
        "sec": dt,
        "events": len(h["events"]),
    }


def main():
    print("=" * 72)
    print("Marvis MoE v5 真实训练：MNIST（CPU）")
    print("=" * 72)
    Xtr, ytr = load_mnist("train")
    Xte, yte = load_mnist("t10k")
    print(f"数据: train={Xtr.shape[0]} test={Xte.shape[0]} dim={Xtr.shape[1]}")

    Xtr, ytr = sample(Xtr, ytr, 8000, seed=2026)   # 8000 训练样本
    Xte, yte = sample(Xte, yte, 2000, seed=2026)   # 2000 验证样本
    Xtr, Xte = apply_pca(Xtr, Xte, n_components=64)  # 784 -> 64 维（阈值突破）

    d = Xtr.shape[1]          # 64
    E, S, L, K, topk = 64, 2, 2, 10, 6
    cfg = Config(d=d, h=256, E=E, S=S, L=L, topk=topk, out_dim=K)
    print(f"架构: E={E} 路由专家, S={S} 共享专家, L={L} 层, topk={topk}, 参数量可数")
    n_params = sum(p.numel() for p in MarvisMoE(cfg).parameters())
    print(f"参数量: {n_params:,}")

    results = {}
    for gov in (False, True):
        print(f"\n>>> 训练中：治理={'开' if gov else '关'}  (epochs=4, batch=256, lr=1e-3)")
        r = run_gov(gov, cfg, Xtr, ytr, Xte, yte, epochs=4, batch_size=256, lr=1e-3)
        print(f"  训练正确率: {r['train_acc']:.4f} | 测试正确率: {r['test_acc']:.4f}")
        print(f"  末段损失  : {r['loss']:.4f} | 负载std: {r['std']:.4f} | 活跃: {r['active']:.2%}")
        print(f"  耗时      : {r['sec']:.1f}s | 治理事件: {r['events']} 次")
        results[gov] = r

    print("\n" + "=" * 72)
    print("治理开/关公平对比（同 seed=2026，同数据子集）")
    print("=" * 72)
    print(f"{'指标':<14}{'治理关':>14}{'治理开':>14}")
    for name, key in [("测试正确率", "test_acc"), ("训练正确率", "train_acc"),
                      ("末段损失", "loss"), ("负载std", "std"), ("活跃专家比", "active")]:
        print(f"{name:<14}{results[False][key]:>14.4f}{results[True][key]:>14.4f}")


if __name__ == "__main__":
    main()
