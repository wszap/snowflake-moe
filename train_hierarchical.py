# -*- coding: utf-8 -*-
"""Hierarchical CellMoE —— 两级细胞化架构训练脚本（字符级莎士比亚 LM）

复用 train_lm 的数据管线（load_shakespeare / lm_batch / eval_ppl），
模型为 snowflake_moe.HierarchicalCellMoE_LM（4 细胞 + 上层路由，L=2）。

监控指标（沿用单级 + 新增第二级）：
- 忆点利用率：细胞层忆点检索 argmax 覆盖数 / n_memory（4 细胞平均）
- 细胞器利用率：细胞层 Top-K 激活去重数 / n_organelles（4 细胞平均）
- 组装多样性：细胞层熵（nat，平均）
- 第二级路由利用率：gate argmax 去重覆盖数 / n_cells
- 第二级 gate 熵：路由分散程度（nat，越大越分散）

用法：
    python train_hierarchical.py --smoke             # 冒烟：1 batch forward/backward
    python train_hierarchical.py --run --seed 2026   # 训练 5 epoch + 验证 + 落 CSV
"""
import argparse
import csv
import gc
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from snowflake_moe import HierarchicalCellMoE_LM  # noqa: E402
from train_lm import load_shakespeare, lm_batch, eval_ppl  # noqa: E402
from marvis_moe import Config  # noqa: E402

OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output", "results_snowflake_10001.csv"))
CSV_HEADER = ["data", "seed", "mode", "epochs", "val_ppl", "val_loss",
              "train_loss", "mem_util", "org_util", "assemble_entropy",
              "route_util", "gate_entropy",
              "params", "fixed_params", "param_ratio", "sec"]

# fixed MoE 对比基线（与优化后 PPL=12.85 同配置：d=64, E=4, S=1, L=2, h=128）
FIXED_CFG = dict(d=64, h=128, E=4, S=1, L=2, topk=2)

# ---- 机制修正（沿用 v3 单级设定，不调参）----
# 本次唯一变更：层级嵌套（4×CellMoE + 上层路由）；不改组装头/忆点
LAMBDA_ENT = 0.05          # 熵正则权重：loss = ce - lambda_ent * entropy
LAMBDA_MEM = 0.05          # 忆点覆盖正则权重：+ lambda_mem * (usage^2).sum() * n_memory
LAMBDA_ORG = 0.05          # 细胞器均衡 aux 权重：+ lambda_org * (freq_soft^2).sum() * n_org
CONFIG_NOTES = (
    "两级细胞化 10001：4×CellMoE(8 细胞器/32 忆点/topk4, 纯线性组装=v3 结构) + 上层路由"
    "(Gumbel-Softmax 软聚合, topk_cell=2 仅作验收口径)；L=2；lambda/lr/batch 沿用 v3；"
    "验证核心问题：两级嵌套（细胞→组织）是否有性能增益；"
    "第二级监控：route_util(gate argmax 覆盖/n_cells), gate_entropy(gate 熵)"
)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def monitor_stats_hier(info, n_cells, n_organelles, n_memory):
    """两级监控。info: {'gate': [N, n_cells], 'cell_infos': [n_cells × dict]}"""
    gate = info['gate']
    cell_infos = info['cell_infos']
    mems, orgs, ents = [], [], []
    for ci in cell_infos:
        weights = ci['weights']
        attn = ci['memory_attn']
        topk_idx = ci['topk_idx']
        org_used = len(torch.unique(topk_idx).tolist()) / n_organelles
        mem_used = len(torch.unique(attn.argmax(-1)).tolist()) / n_memory
        eps = 1e-8
        ent_attn = (-(attn * torch.log(attn + eps)).sum(-1)).mean().item()
        ent_w = (-(weights * torch.log(weights + eps)).sum(-1)).mean().item()
        mems.append(mem_used)
        orgs.append(org_used)
        ents.append((ent_attn + ent_w) / 2.0)
    mem_used = float(np.mean(mems))
    org_used = float(np.mean(orgs))
    ent = float(np.mean(ents))
    # 第二级：路由利用率 + gate 熵
    route_used = len(torch.unique(gate.argmax(-1)).tolist()) / n_cells
    eps = 1e-8
    gate_ent = (-(gate * torch.log(gate + eps)).sum(-1)).mean().item()
    return mem_used, org_used, ent, route_used, gate_ent


def train_hierarchical(args):
    global OUT_CSV
    if getattr(args, "out", None):
        OUT_CSV = os.path.abspath(args.out)
    set_seed(args.seed)
    vocab_size, train_ids, val_ids, chars = load_shakespeare()
    model = HierarchicalCellMoE_LM(d=args.d, vocab_size=vocab_size,
                                   n_cells=args.n_cells,
                                   n_organelles=args.n_organelles,
                                   n_memory=args.n_memory,
                                   topk_organelle=args.topk_organelle,
                                   topk_cell=args.topk_cell,
                                   L=args.L).to(DEVICE)
    params = count_params(model)

    # fixed MoE 同接口参数量（构造但不训练）
    from train_lm import MarvisMoE_LM
    fixed_model = MarvisMoE_LM(Config(**FIXED_CFG), vocab_size)
    fixed_params = count_params(fixed_model)
    del fixed_model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_steps_per_epoch = max(1, train_ids.numel() // (args.seq_len * args.batch_size))
    total_steps = n_steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * 0.05))

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        return args.lr

    if args.smoke:
        xb, yb = lm_batch(train_ids, args.batch_size, args.seq_len, seed=args.seed)
        logits, info = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        loss.backward()
        mem_used, org_used, ent, route_used, gate_ent = monitor_stats_hier(
            info, args.n_cells, args.n_organelles, args.n_memory)
        print(f"[SMOKE] ok  loss={loss.item():.4f}  mem_util={mem_used:.3f} "
              f"org_util={org_used:.3f}  entropy={ent:.3f} "
              f"route_util={route_used:.3f}  gate_ent={gate_ent:.3f}")
        print(f"[SMOKE] params={params}  fixed_params={fixed_params}  "
              f"ratio={params / fixed_params:.3f}")
        return

    t0 = time.time()
    print(f"[CONFIG_NOTES] {CONFIG_NOTES}")
    train_loss_sum, train_steps = 0.0, 0
    mem_util_acc, org_util_acc, ent_acc = [], [], []
    route_acc, gate_acc = [], []
    step_global = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_sum, epoch_steps = 0.0, 0
        for _ in range(n_steps_per_epoch):
            xb, yb = lm_batch(train_ids, args.batch_size, args.seq_len,
                              seed=args.seed + step_global)
            for g in opt.param_groups:
                g['lr'] = lr_at(step_global)
            opt.zero_grad(set_to_none=True)
            logits, info = model(xb)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            # ---- 细胞层正则：拼接 4 细胞的分布（沿用 v3 三项）----
            w = torch.cat([ci['weights'] for ci in info['cell_infos']], dim=0)
            attn = torch.cat([ci['memory_attn'] for ci in info['cell_infos']], dim=0)
            entropy = -(w * torch.log(w + 1e-9)).sum(-1).mean()
            memory_usage = attn.mean(dim=0)                 # [n_memory]
            aux_memory = (memory_usage * memory_usage).sum() * args.n_memory
            freq_soft = w.mean(dim=0)                       # [n_org]
            aux_organelle = (freq_soft * freq_soft).sum() * args.n_organelles
            loss = (ce_loss - LAMBDA_ENT * entropy
                    + LAMBDA_MEM * aux_memory + LAMBDA_ORG * aux_organelle)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step_global += 1

            mem_used, org_used, ent, route_used, gate_ent = monitor_stats_hier(
                info, args.n_cells, args.n_organelles, args.n_memory)
            mem_util_acc.append(mem_used)
            org_util_acc.append(org_used)
            ent_acc.append(ent)
            route_acc.append(route_used)
            gate_acc.append(gate_ent)
            train_loss_sum += loss.item()
            train_steps += 1
            epoch_loss_sum += loss.item()
            epoch_steps += 1
            del logits, loss
        avg_ce, ppl = eval_ppl(model, val_ids, vocab_size, args.seq_len,
                               n_batch=20, batch_size=16)
        print(f"[EPOCH {epoch}/{args.epochs}] train_loss={epoch_loss_sum / epoch_steps:.4f} "
              f"val_ppl={ppl:.3f}  ({(time.time() - t0):.0f}s)")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    sec = time.time() - t0
    train_loss = train_loss_sum / train_steps
    avg_ce, val_ppl = eval_ppl(model, val_ids, vocab_size, args.seq_len,
                               n_batch=20, batch_size=16)
    mem_util = float(np.mean(mem_util_acc))
    org_util = float(np.mean(org_util_acc))
    ent = float(np.mean(ent_acc))
    route_util = float(np.mean(route_acc))
    gate_entropy = float(np.mean(gate_acc))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_HEADER)
        w.writerow(["shakespeare", args.seed, "snowflake_10001", args.epochs,
                    round(val_ppl, 4), round(avg_ce, 4), round(train_loss, 4),
                    round(mem_util, 4), round(org_util, 4), round(ent, 4),
                    round(route_util, 4), round(gate_entropy, 4),
                    params, fixed_params, round(params / fixed_params, 4),
                    round(sec, 1)])
    print(f"[DONE] val_ppl={val_ppl:.3f}  mem_util={mem_util:.3f}  "
          f"org_util={org_util:.3f}  entropy={ent:.3f}  "
          f"route_util={route_util:.3f}  gate_ent={gate_entropy:.3f}  sec={sec:.0f}")
    print(f"[DONE] params={params}  fixed_params={fixed_params}  "
          f"ratio={params / fixed_params:.3f}")
    print(f"[DONE] CSV -> {OUT_CSV}")


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n_cells", type=int, default=4)
    ap.add_argument("--n_organelles", type=int, default=8)
    ap.add_argument("--n_memory", type=int, default=32)
    ap.add_argument("--topk_organelle", type=int, default=4)
    ap.add_argument("--topk_cell", type=int, default=2)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=str, default="", help="覆盖输出 CSV 路径")
    args = ap.parse_args()

    if not (args.smoke or args.run):
        ap.error("请指定 --smoke 或 --run")
    train_hierarchical(args)
