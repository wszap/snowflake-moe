# -*- coding: utf-8 -*-
"""Snowflake MoE —— 细胞化架构训练脚本（字符级莎士比亚 LM）

复用 train_lm 的数据管线（load_shakespeare / lm_batch / eval_ppl），
模型为 snowflake_moe.SnowflakeMoE_LM（无治理机制，纯前向训练）。

监控指标：
- 忆点利用率：每 batch 忆点检索分布 argmax 覆盖的忆点数 / n_memory
- 细胞器利用率：Top-K 实际激活的细胞器去重数 / n_organelles
- 组装多样性：忆点检索分布与组装权重分布的熵（nat，越大越分散）

用法：
    python train_snowflake.py --smoke             # 冒烟：1 batch forward/backward
    python train_snowflake.py --run               # 训练 5 epoch + 验证 + 落 CSV
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

from snowflake_moe import SnowflakeMoE_LM  # noqa: E402
from train_lm import load_shakespeare, lm_batch, eval_ppl  # noqa: E402
from marvis_moe import Config  # noqa: E402

OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output", "results_snowflake_00005.csv"))
CSV_HEADER = ["data", "seed", "mode", "epochs", "val_ppl", "val_loss",
              "train_loss", "mem_util", "org_util", "assemble_entropy",
              "params", "fixed_params", "param_ratio", "sec"]

# fixed MoE 对比基线（与优化后 PPL=12.85 同配置：d=64, E=4, S=1, L=2, h=128）
FIXED_CFG = dict(d=64, h=128, E=4, S=1, L=2, topk=2)

# ---- 机制修正（非调参，属防坍缩/信息效率设计补强）----
# 00005: 组装头改残差+零初始化（AssemblyHead: x + fc2(SiLU(fc1(x)))，fc2 零初始化，
#      epoch0 恒等输出=纯线性，训练只学残差，不破坏线性先验）
LAMBDA_ENT = 0.05          # 熵正则权重：loss = ce - lambda_ent * entropy
LAMBDA_MEM = 0.05          # 忆点覆盖正则权重：+ lambda_mem * (usage^2).sum() * n_memory
LAMBDA_ORG = 0.05          # 细胞器均衡 aux 权重：+ lambda_org * (freq_soft^2).sum() * n_org
CONFIG_NOTES = (
    "版本命名规则：ABBBB（A=大版本，BBBB=小版本）；当前：00005（v5 残差+零初始化实验）；"
    "下次架构级变更：10001；"
    "机制修正：组装头残差+零初始化 00005——①00004(MLP组装头)诊断：编码器/检索正常、PPL=14.463、"
    "org_util 29.5% 劣化，MLP 自由度未收敛破坏稀疏；②00005 改 AssemblyHead=fc1(20,40)+SiLU+fc2(40,20)"
    "，fc2 weight/bias 零初始化，forward=x+fc2(act(fc1(x)))，epoch0 恒等=纯线性(00003)行为；"
    "③不调 lambda、不改 batch/lr；不占调参名额"
)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def monitor_stats(info, n_organelles, n_memory, device=DEVICE):
    """从一层 forward 的监控信息计算利用率/熵。info: dict。"""
    weights = info['weights']        # [N, n_organelles]
    attn = info['memory_attn']       # [N, n_memory]
    topk_idx = info['topk_idx']      # [N, topk]
    n = weights.shape[0]
    # 细胞器利用率：Top-K 去重覆盖数 / 总数
    org_used = len(torch.unique(topk_idx).tolist()) / n_organelles
    # 忆点利用率：检索 argmax 去重覆盖数 / 总数
    mem_used = len(torch.unique(attn.argmax(-1)).tolist()) / n_memory
    # 组装多样性：两类分布熵的均值（nat）
    eps = 1e-8
    ent_attn = (-(attn * torch.log(attn + eps)).sum(-1)).mean().item()
    ent_w = (-(weights * torch.log(weights + eps)).sum(-1)).mean().item()
    return mem_used, org_used, (ent_attn + ent_w) / 2.0


def diagnose_representations(model, val_ids, seq_len, device=DEVICE, seed=2026,
                             batch_size=128):
    """机制诊断：验证集 1 batch，检查编码器坍缩与忆点检索坍缩。"""
    model.eval()
    xb, yb = lm_batch(val_ids, batch_size, seq_len, device, seed=seed)
    x = model.embed(xb)
    x = model.in_proj(x)
    B, T, d = x.shape
    xr = x.reshape(B * T, d)
    with torch.no_grad():
        for li, layer in enumerate(model.layers):
            v = layer.norm(layer.encoder(xr))                # [N, d]
            sim = v @ layer.memory_keys.T / (d ** 0.5)
            attn = F.softmax(sim / 1.0, -1)                  # eval 口径
            v_dim_var = v.std(dim=0).mean().item()           # 维度间方差
            # 采样 500 个样本计算余弦相似度均值（去对角线）
            idx = torch.randperm(v.shape[0], device=v.device)[:500]
            vn = F.normalize(v[idx], dim=-1)
            cos = vn @ vn.T
            m = ~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)
            cos_mean = cos[m].mean().item()
            n_mem = len(torch.unique(attn.argmax(-1)).tolist())
            enc_state = ("编码器坍缩" if cos_mean > 0.9
                         else "编码器正常" if cos_mean < 0.5 else "编码器边界")
            ret_state = ("检索坍缩" if n_mem < 5
                         else "检索正常" if n_mem > 20 else "检索边界")
            print(f"[DIAG layer{li}] v_dim_var={v_dim_var:.4f} "
                  f"cos_mean={cos_mean:.4f} attn_argmax_mem={n_mem}")
            print(f"[DIAG layer{li}] 判定: 编码器->{enc_state}, 检索->{ret_state}")
    model.train()
    return


def load_data(args):
    """按 --data 加载数据；留空则用莎士比亚（原行为）。支持 .pt dict:
    {'train': LongTensor, 'val': LongTensor, 'vocab_size': int, 'vocab': dict}"""
    data_path = getattr(args, "data", None)
    if data_path:
        p = data_path if os.path.isabs(data_path) else os.path.join(BASE, data_path)
        if not os.path.exists(p):
            raise FileNotFoundError(f"data file not found: {p}")
        d = torch.load(p, map_location="cpu")
        vocab_size = d["vocab_size"]
        train_ids = d["train"].long()
        val_ids = d["val"].long()
        chars = d.get("vocab", {})
        return vocab_size, train_ids, val_ids, chars, os.path.basename(p)
    vocab_size, train_ids, val_ids, chars = load_shakespeare()
    return vocab_size, train_ids, val_ids, chars, "shakespeare"


def train_snowflake(args):
    global OUT_CSV
    if getattr(args, "out", None):
        OUT_CSV = os.path.abspath(args.out)
    set_seed(args.seed)
    vocab_size, train_ids, val_ids, chars, data_name = load_data(args)
    model = SnowflakeMoE_LM(d=args.d, vocab_size=vocab_size,
                            n_organelles=args.n_organelles,
                            n_memory=args.n_memory,
                            topk_organelle=args.topk,
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
        mem_used, org_used, ent = monitor_stats(info, args.n_organelles, args.n_memory)
        print(f"[SMOKE] ok  loss={loss.item():.4f}  mem_util={mem_used:.3f} "
              f"org_util={org_used:.3f}  entropy={ent:.3f}")
        print(f"[SMOKE] params={params}  fixed_params={fixed_params}  "
              f"ratio={params / fixed_params:.3f}")
        return

    t0 = time.time()
    print(f"[CONFIG_NOTES] {CONFIG_NOTES}")
    train_loss_sum, train_steps = 0.0, 0
    mem_util_acc, org_util_acc, ent_acc = [], [], []
    step_global = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_sum, epoch_steps = 0.0, 0
        for _ in range(n_steps_per_epoch):
            xb, yb = lm_batch(train_ids, args.batch_size, args.seq_len, seed=args.seed + step_global)
            for g in opt.param_groups:
                g['lr'] = lr_at(step_global)
            opt.zero_grad(set_to_none=True)
            logits, info = model(xb)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            # ---- 机制修正：组装熵正则（防坍缩）----
            w = info['weights']                                   # [N, n_org]
            entropy = -(w * torch.log(w + 1e-9)).sum(-1).mean()
            # ---- 机制修正：忆点覆盖正则（可导）----
            memory_usage = info['memory_attn'].mean(dim=0)        # [n_memory]
            aux_memory = (memory_usage * memory_usage).sum() * args.n_memory
            # ---- 机制修正：细胞器均衡 aux（可导版，替代 bincount）----
            freq_soft = w.mean(dim=0)                             # [n_org]
            aux_organelle = (freq_soft * freq_soft).sum() * args.n_organelles
            loss = (ce_loss - LAMBDA_ENT * entropy
                    + LAMBDA_MEM * aux_memory + LAMBDA_ORG * aux_organelle)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step_global += 1

            mem_used, org_used, ent = monitor_stats(info, args.n_organelles, args.n_memory)
            mem_util_acc.append(mem_used)
            org_util_acc.append(org_used)
            ent_acc.append(ent)
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

    if getattr(args, "diagnose", False):
        print("[DIAG] 训练完成，对验证集 1 batch 做机制诊断（不写 CSV）")
        diagnose_representations(model, val_ids, args.seq_len, device=DEVICE,
                                 seed=args.seed, batch_size=args.batch_size)
        return

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_HEADER)
        w.writerow(["shakespeare", args.seed, "snowflake_00005", args.epochs,
                    round(val_ppl, 4), round(avg_ce, 4), round(train_loss, 4),
                    round(mem_util, 4), round(org_util, 4), round(ent, 4),
                    params, fixed_params, round(params / fixed_params, 4),
                    round(sec, 1)])
    print(f"[DONE] val_ppl={val_ppl:.3f}  mem_util={mem_util:.3f}  "
          f"org_util={org_util:.3f}  entropy={ent:.3f}  sec={sec:.0f}")
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
    ap.add_argument("--data", type=str, default="",
                    help="训练数据 .pt 文件（dict: train/val/vocab_size），留空用莎士比亚")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n_organelles", type=int, default=20)
    ap.add_argument("--n_memory", type=int, default=64)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=str, default="", help="覆盖输出 CSV 路径")
    ap.add_argument("--diagnose", action="store_true",
                    help="训练完成后打印编码器/检索诊断，不写 CSV")
    args = ap.parse_args()

    if not (args.smoke or args.run):
        ap.error("请指定 --smoke 或 --run")
    train_snowflake(args)
