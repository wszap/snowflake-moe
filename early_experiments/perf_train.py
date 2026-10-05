# -*- coding: utf-8 -*-
"""perf_train.py —— v3_base_298k（HierarchicalCellMoE_LM）性能极致化实验脚本

与 train_hierarchical.py 完全一致的训练语义（loss 三项正则 / warmup / clip /
monitor_stats_hier），仅叠加性能优化开关，保证 PPL 可比：
  --dl        DataLoader(num_workers=4, prefetch_factor=2, pin_memory=True)
  --item10    loss.item() 每 10 步一次（累积用 loss.detach() GPU 加法）
  --bench     torch.backends.cudnn.benchmark = True
  --amp       torch.autocast(fp16) + GradScaler
  --compile   torch.compile(mode="reduce-overhead")
  --vec       CellMoE.forward 器官循环向量化补丁（批量 einsum，数学等价）
监控：每 10 步 torch.cuda.utilization() / psutil.cpu_percent() / 数据加载耗时占比；
每 epoch 检查 GPU 温度，>=80°C 暂停 600s。
输出：temp/perf_results.csv 追加一行（tag, ppl, train_loss, sec, gpu_util,
cpu_util, data_ratio, steps_per_sec）
"""
import argparse
import contextlib
import csv
import gc
import math
import os
import subprocess
import sys
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import snowflake_moe as sm  # noqa: E402
from train_lm import load_shakespeare, lm_batch, eval_ppl  # noqa: E402

RESULT_CSV = os.path.join(BASE, "perf_results.csv")
LAMBDA_ENT = 0.05
LAMBDA_MEM = 0.05
LAMBDA_ORG = 0.05
TEMP_LIMIT = 80          # °C 红线
TEMP_SLEEP = 600         # s


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def check_gpu_temp():
    """GPU 温度红线：>=80°C 暂停 10 分钟。返回当前温度。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"])
        temp = int(out.decode().strip().split()[0])
    except Exception:
        return -1
    if temp >= TEMP_LIMIT:
        print(f"[TEMP] GPU={temp}C 超过红线 {TEMP_LIMIT}C，暂停 {TEMP_SLEEP}s ...")
        time.sleep(TEMP_SLEEP)
    return temp


def monitor_stats_hier(info, n_cells, n_organelles, n_memory):
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
    route_used = len(torch.unique(gate.argmax(-1)).tolist()) / n_cells
    eps = 1e-8
    gate_ent = (-(gate * torch.log(gate + eps)).sum(-1)).mean().item()
    return (float(np.mean(mems)), float(np.mean(orgs)), float(np.mean(ents)),
            route_used, gate_ent)


def make_model(d, vocab_size, n_cells, n_organelles, n_memory,
               topk_organelle, topk_cell, L):
    model = sm.HierarchicalCellMoE_LM(d=d, vocab_size=vocab_size,
                                      n_cells=n_cells, n_organelles=n_organelles,
                                      n_memory=n_memory,
                                      topk_organelle=topk_organelle,
                                      topk_cell=topk_cell, L=L).to(DEVICE)
    return model


def vectorized_cell_forward(self, x):
    """CellMoE.forward 向量化版：器官循环 torch.stack([o(x)...]) → 批量 einsum。

    数学等价：y = W2_i @ SiLU(W1_i @ x)，参数仍来自 self.organelles ModuleList。
    """
    v = self.norm(self.encoder(x))
    if self.new_memory_keys is None:
        keys = self.memory_keys
        asm = self.memory_assembly
    else:
        keys = torch.cat([self.memory_keys, self.new_memory_keys], dim=0)
        asm = torch.cat([self.memory_assembly, self.new_memory_assembly], dim=0)
    sim = v @ keys.T / (self.d ** 0.5)
    if self.training:
        attn = F.gumbel_softmax(sim, tau=1.0, hard=False, dim=-1)
    else:
        attn = F.softmax(sim / 1.0, dim=-1)
    assembly = attn @ asm
    weights = F.softmax(assembly, dim=-1)
    topk_w, topk_idx = torch.topk(weights, self.topk, dim=-1)
    topk_w = topk_w / topk_w.sum(-1, keepdim=True)
    # ---- 向量化器官：堆叠权重 → einsum（等价 o(x) 逐器官调用）----
    W1 = torch.stack([o.net[0].weight for o in self.organelles], dim=0)  # [n_org, h, d]
    W2 = torch.stack([o.net[2].weight for o in self.organelles], dim=0)  # [n_org, d, h]
    h1 = F.silu(torch.einsum('ohd,bd->boh', W1, x))                       # [B, n_org, h]
    org_out = torch.einsum('odh,boh->bod', W2, h1)                        # [B, n_org, d]
    w3 = topk_w.unsqueeze(-1)
    idx3 = topk_idx.unsqueeze(-1).expand(-1, -1, self.d)
    out = (org_out.gather(1, idx3) * w3).sum(1)
    out = self.head(out)
    return out, {'weights': weights, 'memory_attn': attn, 'topk_idx': topk_idx}


def _identity_collate(batch):
    """batch_size=None 时 collate 应用于单个 item：原样返回 (xb, yb)（可 pickle）。"""
    return batch


class RandomBatchIterDataset(torch.utils.data.IterableDataset):
    """无限随机 batch 流：每 worker 独立确定性采样（seed 与 lm_batch 同构）。"""
    def __init__(self, ids, batch_size, seq_len):
        self.ids = ids
        self.batch_size = batch_size
        self.seq_len = seq_len

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        n_workers = worker.num_workers if worker is not None else 1
        i = 0
        while True:
            seed = 2026 + wid * 100000 + i
            xb, yb = lm_batch(self.ids, self.batch_size, self.seq_len,
                              device="cpu", seed=seed)
            yield xb, yb
            i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--dl", action="store_true", help="DataLoader 多 worker")
    ap.add_argument("--item10", action="store_true", help="loss.item 每 10 步")
    ap.add_argument("--bench", action="store_true", help="cudnn.benchmark")
    ap.add_argument("--amp", action="store_true", help="AMP fp16 + GradScaler")
    ap.add_argument("--compile", action="store_true", help="torch.compile")
    ap.add_argument("--vec", action="store_true", help="器官循环向量化")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no_temp_wait", action="store_true")
    args = ap.parse_args()

    # ---- v3_base_298k 配置（与 checkpoints/v3_base_298k.pt 的 cfg 一致）----
    d, vocab_size = 64, 65
    n_cells, n_organelles, n_memory = 4, 6, 24
    topk_organelle, topk_cell, L = 4, 2, 2
    seq_len, batch_size, lr = 64, 128, 3e-4
    epochs = args.epochs

    if args.bench:
        torch.backends.cudnn.benchmark = True

    set_seed(args.seed)
    _, train_ids, val_ids, _ = load_shakespeare()
    model = make_model(d, vocab_size, n_cells, n_organelles, n_memory,
                       topk_organelle, topk_cell, L)

    if args.vec:
        sm.CellMoE.forward = vectorized_cell_forward
        print("[VEC] 已打器官循环向量化补丁（批量 einsum，数学等价）")

    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[COMPILE] torch.compile 成功")
        except Exception as e:
            print(f"[COMPILE] 失败，回滚: {e}")
            model = make_model(d, vocab_size, n_cells, n_organelles, n_memory,
                               topk_organelle, topk_cell, L)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_steps_per_epoch = max(1, train_ids.numel() // (seq_len * batch_size))
    total_steps = n_steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * 0.05))

    def lr_at(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        return lr

    # ---- 数据管线 ----
    if args.dl:
        ds = RandomBatchIterDataset(train_ids, batch_size, seq_len)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=4, prefetch_factor=2,
            pin_memory=True, collate_fn=_identity_collate)
        data_iter = iter(dl)

        def next_batch():
            xb, yb = next(data_iter)
            return (xb.to(DEVICE, non_blocking=True),
                    yb.to(DEVICE, non_blocking=True))
    else:
        def next_batch():
            return lm_batch(train_ids, batch_size, seq_len, device=DEVICE,
                            seed=args.seed + step_global)

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    amp_ctx = (torch.autocast("cuda", dtype=torch.float16) if args.amp
               else contextlib.nullcontext())

    # ---- 监控累计 ----
    gpu_utils, cpu_utils = [], []
    t_data_sum, t_calc_sum = 0.0, 0.0

    t0 = time.time()
    train_loss_sum, train_steps = 0.0, 0
    mem_acc, org_acc, ent_acc, route_acc, gate_acc = [], [], [], [], []
    step_global = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for _ in range(n_steps_per_epoch):
            for g in opt.param_groups:
                g['lr'] = lr_at(step_global)

            t_data0 = time.perf_counter()
            xb, yb = next_batch()
            t_data1 = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            with amp_ctx:
                logits, info = model(xb)
                ce_loss = F.cross_entropy(logits.view(-1, vocab_size),
                                          yb.view(-1))
                w = torch.cat([ci['weights'] for ci in info['cell_infos']], dim=0)
                attn = torch.cat([ci['memory_attn'] for ci in info['cell_infos']], dim=0)
                entropy = -(w * torch.log(w + 1e-9)).sum(-1).mean()
                memory_usage = attn.mean(dim=0)
                aux_memory = (memory_usage * memory_usage).sum() * n_memory
                freq_soft = w.mean(dim=0)
                aux_organelle = (freq_soft * freq_soft).sum() * n_organelles
                loss = (ce_loss - LAMBDA_ENT * entropy
                        + LAMBDA_MEM * aux_memory + LAMBDA_ORG * aux_organelle)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step_global += 1

            mem_used, org_used, ent, route_used, gate_ent = monitor_stats_hier(
                info, n_cells, n_organelles, n_memory)
            mem_acc.append(mem_used)
            org_acc.append(org_used)
            ent_acc.append(ent)
            route_acc.append(route_used)
            gate_acc.append(gate_ent)

            if args.item10:
                train_loss_sum = train_loss_sum + loss.detach()
            else:
                train_loss_sum = train_loss_sum + loss.item()
            train_steps += 1

            t_data_sum += (t_data1 - t_data0)
            t_calc_sum += (time.perf_counter() - t_data1)

            if (step_global % 10) == 0:
                gpu_u = torch.cuda.utilization(DEVICE) if DEVICE == "cuda" else 0
                cpu_u = psutil.cpu_percent(interval=0.1)
                gpu_utils.append(gpu_u)
                cpu_utils.append(cpu_u)
                cur = (train_loss_sum / train_steps).item() if args.item10 \
                    else train_loss_sum / train_steps
                print(f"   step {step_global}/{total_steps} loss={cur:.4f} "
                      f"gpu={gpu_u:.0f}% cpu={cpu_u:.0f}% "
                      f"data={t_data_sum / (t_data_sum + t_calc_sum):.1%} "
                      f"({(time.time() - t0):.0f}s)")
            del logits, loss
            if (step_global % 100) == 0 and not args.no_temp_wait:
                check_gpu_temp()
        # epoch 末
        if not args.no_temp_wait:
            check_gpu_temp()
        with torch.no_grad():
            avg_ce, ppl = eval_ppl(model, val_ids, vocab_size, seq_len,
                                   n_batch=20, batch_size=16)
        print(f"[EPOCH {epoch}/{epochs}] val_ppl={ppl:.3f} "
              f"({(time.time() - t0):.0f}s)")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    sec = time.time() - t0
    avg_ce, val_ppl = eval_ppl(model, val_ids, vocab_size, seq_len,
                               n_batch=20, batch_size=16)
    train_loss = (train_loss_sum / train_steps).item() if args.item10 \
        else train_loss_sum / train_steps
    gpu_util = float(np.mean(gpu_utils)) if gpu_utils else 0.0
    cpu_util = float(np.mean(cpu_utils)) if cpu_utils else 0.0
    data_ratio = t_data_sum / max(1e-9, (t_data_sum + t_calc_sum))
    steps_per_sec = total_steps / sec

    note = "+".join(k for k, v in
                    [("dl", args.dl), ("item10", args.item10),
                     ("bench", args.bench), ("amp", args.amp),
                     ("compile", args.compile), ("vec", args.vec)] if v) or "-"

    os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
    new_file = not os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["tag", "ppl", "train_loss", "sec", "gpu_util",
                        "cpu_util", "data_ratio", "steps_per_sec", "note"])
        w.writerow([args.tag, round(val_ppl, 4), round(train_loss, 4),
                    round(sec, 1), round(gpu_util, 1), round(cpu_util, 1),
                    round(data_ratio, 4), round(steps_per_sec, 2), note])

    print(f"[DONE] tag={args.tag} ppl={val_ppl:.3f} sec={sec:.0f} "
          f"gpu_util={gpu_util:.0f}% cpu_util={cpu_util:.0f}% "
          f"data_ratio={data_ratio:.1%} steps/s={steps_per_sec:.1f} "
          f"note={note}")


if __name__ == "__main__":
    main()
