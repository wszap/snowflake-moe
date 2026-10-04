# -*- coding: utf-8 -*-
"""阶段五：TinyStories 100MB 扩展性验证
- 数据：tinystories_100mb.txt（TinyStories-train.txt 前 100MB），字符级，seq128，90/10
- CellMoE: ImprovedHierarchicalCellMoE_LM(d=128, n_cells=4, n_organelles=8,
          n_memory=32, topk_organelle=4, topk_cell=2, L=4, gate_ent_reg=True)
- Fixed: dense FFN transformer（d=128, L=4, 每层 FFN 宽 8d, SiLU+残差+LayerNorm）
- 各 5 epoch, seed 2026, batch=1024, lr=3e-4, warmup 5%, clip 1.0
- 输出 output/results_tinystories.csv
- 验收: CellMoE PPL <= Fixed*1.05；GPU 利用率 >60%（训练采样）
"""
import csv
import gc
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from snowflake_moe_improved import ImprovedHierarchicalCellMoE_LM  # noqa: E402

SEQ_LEN, BATCH_SIZE, EPOCHS, LR, SEED = 128, 512, 5, 3e-4, 2026
LAMBDA_ENT, LAMBDA_MEM, LAMBDA_ORG = 0.05, 0.05, 0.05
DATA = os.path.join(BASE, "tinystories_100mb.txt")
OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output",
                                       "results_tinystories.csv"))


# ---------------- Fixed dense FFN transformer ----------------
class FFNBlock(nn.Module):
    def __init__(self, d, width=8):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, width * d)
        self.fc2 = nn.Linear(width * d, d)

    def forward(self, x):
        h = F.silu(self.fc1(self.norm(x)))
        return self.fc2(h) + x


class FixedFFN_LM(nn.Module):
    """dense 固定基线：d128, L4 层, 每层 FFN 宽 8d（E8 全激活）。"""

    def __init__(self, d, vocab_size, L=4, width=8):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d)
        self.in_proj = nn.Linear(d, d, bias=False)
        self.layers = nn.ModuleList([FFNBlock(d, width) for _ in range(L)])
        self.head = nn.Linear(d, vocab_size)

    def forward(self, tokens):
        x = self.in_proj(self.embed(tokens))
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ---------------- 数据 ----------------
def load_tinystories(path=DATA, val_frac=0.1, seed=2026):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = int(data.numel() * val_frac)
    rng = random.Random(seed)
    val_start = rng.randint(0, data.numel() - n_val - 1)
    val_ids = data[val_start:val_start + n_val]
    train_ids = torch.cat([data[:val_start], data[val_start + n_val:]])
    return vocab_size, train_ids, val_ids, chars


def lm_batch(ids, batch_size, seq_len, seed):
    g = torch.Generator().manual_seed(seed)
    n = ids.numel() - seq_len - 1
    idx = torch.randint(0, n, (batch_size,), generator=g)
    xb = torch.stack([ids[i:i + seq_len] for i in idx])
    yb = torch.stack([ids[i + 1:i + seq_len + 1] for i in idx])
    return xb.to(DEVICE), yb.to(DEVICE)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gpu_util():
    if DEVICE != "cuda":
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().splitlines()[0].replace("%", ""))
    except Exception:
        return None


def eval_full(model, ids, vocab_size, batch_size=128):
    model.eval()
    k = batch_size * SEQ_LEN
    n = ids.numel()
    total_ce, total_n = 0.0, 0
    with torch.no_grad():
        s = 0
        while s + k + 1 <= n:
            x = ids[s:s + k].view(batch_size, SEQ_LEN).to(DEVICE)
            y = ids[s + 1:s + k + 1].view(batch_size, SEQ_LEN).to(DEVICE)
            logits, _ = model(x)
            ce = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
            total_ce += ce * k
            total_n += k
            s += k
    model.train()
    avg_ce = total_ce / total_n
    return avg_ce, math.exp(avg_ce)


def train_one(model, train_ids, val_ids, vocab_size, tag, use_reg):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_steps = max(1, train_ids.numel() // (SEQ_LEN * BATCH_SIZE))
    total_steps = n_steps * EPOCHS
    warmup = max(1, int(total_steps * 0.05))

    def lr_at(st):
        return LR * (st + 1) / warmup if st < warmup else LR

    t0 = time.time()
    step_global = 0
    util_sum, util_n = 0.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        e_sum, e_n = 0.0, 0
        for _ in range(n_steps):
            xb, yb = lm_batch(train_ids, BATCH_SIZE, SEQ_LEN, seed=SEED + step_global)
            for g in opt.param_groups:
                g['lr'] = lr_at(step_global)
            opt.zero_grad(set_to_none=True)
            logits, info = model(xb)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            loss = ce_loss
            if use_reg and isinstance(info, dict):
                cell_infos = info.get('cell_infos', [info])
                ent_sum = mem_sum = org_sum = 0.0
                for ci in cell_infos:
                    w = ci['weights']
                    ent_sum += -(w * torch.log(w + 1e-9)).sum(-1).mean()
                    mu = ci['memory_attn'].mean(dim=0)
                    mem_sum += (mu * mu).sum() * mu.shape[-1]
                    fs = w.mean(dim=0)
                    org_sum += (fs * fs).sum() * w.shape[-1]
                nc = len(cell_infos)
                gate = info['gate']
                gate_ent = -(gate * (gate + 1e-9).log()).sum(-1).mean()
                loss = (ce_loss - LAMBDA_ENT * (ent_sum / nc + 0.25 * gate_ent)
                        + LAMBDA_MEM * (mem_sum / nc) + LAMBDA_ORG * (org_sum / nc))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step_global += 1
            e_sum += loss.item()
            e_n += 1
            if step_global % 25 == 0:
                u = gpu_util()
                if u is not None:
                    util_sum += u
                    util_n += 1
            del logits, loss
        print(f"[S5 {tag} EPOCH {epoch}/{EPOCHS}] train_loss={e_sum / e_n:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    val_ce, val_ppl = eval_full(model, val_ids, vocab_size)
    avg_util = util_sum / util_n if util_n else 0.0
    print(f"[S5 {tag}] FINAL val_ppl={val_ppl:.4f} avg_gpu_util={avg_util:.1f}%")
    return val_ce, val_ppl, avg_util, round(time.time() - t0, 1)


def main():
    set_seed(SEED)
    vocab_size, train_ids, val_ids, chars = load_tinystories()
    print(f"[S5] tinystories 100MB vocab={vocab_size} "
          f"train={train_ids.numel()} val={val_ids.numel()}")

    cell = ImprovedHierarchicalCellMoE_LM(
        d=128, vocab_size=vocab_size, n_cells=4, n_organelles=8, n_memory=32,
        topk_organelle=4, topk_cell=2, L=4, gate_ent_reg=True).to(DEVICE)
    fixed = FixedFFN_LM(d=128, vocab_size=vocab_size, L=4, width=8)
    print(f"[S5] CellMoE params={count_params(cell)} Fixed params={count_params(fixed)}")

    ce, cp, cu, ct = train_one(cell, train_ids, val_ids, vocab_size, "CellMoE", True)
    # CellMoE 训练完释放显存，再训 Fixed
    del cell
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    fixed = fixed.to(DEVICE)
    fe, fp, fu, ft = train_one(fixed, train_ids, val_ids, vocab_size, "Fixed", False)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "val_ce", "val_ppl", "gpu_util_pct",
                    "sec", "fixed_ppl_ratio"])
        w.writerow(["CellMoE", count_params(cell), round(ce, 4), round(cp, 4),
                    round(cu, 1), ct, round(cp / fp, 4)])
        w.writerow(["Fixed", count_params(fixed), round(fe, 4), round(fp, 4),
                    round(fu, 1), ft, 1.0])
    print(f"[S5] CSV -> {OUT_CSV}")

    # 验收
    ppl_ok = cp <= fp * 1.05
    util_ok = cu > 60.0
    print(f"[S5] CellMoE PPL={cp:.4f} vs Fixed={fp:.4f} "
          f"(ratio={cp / fp:.4f}, 要求<=1.05)")
    print(f"[S5] CellMoE GPU util={cu:.1f}% Fixed={fu:.1f}% (要求 CellMoE>60%)")
    if ppl_ok and util_ok:
        print("[S5 PASS] CellMoE 在 TinyStories 上 PPL 不劣于 Fixed×1.05 且 GPU 利用充分")
    else:
        print(f"[S5 FAIL] ppl_ok={ppl_ok} util_ok={util_ok}")


if __name__ == "__main__":
    main()
