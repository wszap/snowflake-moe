# -*- coding: utf-8 -*-
"""任务1.1 保存基线 checkpoint（版本 20001）

配置：4 细胞 × 6 细胞器 × 24 忆点（298K 参数），训练 5 epoch 至 PPL≈12.78，
保存 checkpoints/v3_base_298k.pt（state_dict + cfg + final_ppl）。
损失/正则/超参与 10001_param_aligned 完全一致（不改架构、不调 lambda）。
"""
import gc
import os
import random
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

CKPT_DIR = os.path.join(BASE, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "v3_base_298k.pt")

LAMBDA_ENT = 0.05
LAMBDA_MEM = 0.05
LAMBDA_ORG = 0.05

CFG = dict(d=64, n_cells=4, n_organelles=6, n_memory=24,
           topk_organelle=4, topk_cell=2, L=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    seed = 2026
    set_seed(seed)
    vocab_size, train_ids, val_ids, chars = load_shakespeare()
    model = HierarchicalCellMoE_LM(d=CFG['d'], vocab_size=vocab_size,
                                   n_cells=CFG['n_cells'],
                                   n_organelles=CFG['n_organelles'],
                                   n_memory=CFG['n_memory'],
                                   topk_organelle=CFG['topk_organelle'],
                                   topk_cell=CFG['topk_cell'],
                                   L=CFG['L']).to(DEVICE)
    params = count_params(model)
    print(f"[1.1] 模型参数: {params} (期望≈298049)")

    seq_len, batch_size, epochs = 64, 128, 5
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    n_steps_per_epoch = max(1, train_ids.numel() // (seq_len * batch_size))
    total_steps = n_steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * 0.05))

    def lr_at(step):
        if step < warmup_steps:
            return 3e-4 * (step + 1) / warmup_steps
        return 3e-4

    t0 = time.time()
    step_global = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_sum, epoch_steps = 0.0, 0
        for _ in range(n_steps_per_epoch):
            xb, yb = lm_batch(train_ids, batch_size, seq_len, seed=seed + step_global)
            for g in opt.param_groups:
                g['lr'] = lr_at(step_global)
            opt.zero_grad(set_to_none=True)
            logits, info = model(xb)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            w = torch.cat([ci['weights'] for ci in info['cell_infos']], dim=0)
            attn = torch.cat([ci['memory_attn'] for ci in info['cell_infos']], dim=0)
            entropy = -(w * torch.log(w + 1e-9)).sum(-1).mean()
            memory_usage = attn.mean(dim=0)
            aux_memory = (memory_usage * memory_usage).sum() * CFG['n_memory']
            freq_soft = w.mean(dim=0)
            aux_organelle = (freq_soft * freq_soft).sum() * CFG['n_organelles']
            loss = (ce_loss - LAMBDA_ENT * entropy
                    + LAMBDA_MEM * aux_memory + LAMBDA_ORG * aux_organelle)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step_global += 1
            epoch_loss_sum += loss.item()
            epoch_steps += 1
            del logits, loss
        avg_ce, ppl = eval_ppl(model, val_ids, vocab_size, seq_len,
                               n_batch=20, batch_size=16)
        print(f"[EPOCH {epoch}/{epochs}] train_loss={epoch_loss_sum / epoch_steps:.4f} "
              f"val_ppl={ppl:.3f}  ({(time.time() - t0):.0f}s)")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    avg_ce, val_ppl = eval_ppl(model, val_ids, vocab_size, seq_len,
                               n_batch=20, batch_size=16)
    print(f"[1.1] FINAL val_ppl={val_ppl:.3f}  (期望≈12.78)")
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "cfg": dict(CFG, vocab_size=vocab_size),
                "final_ppl": val_ppl, "epochs": epochs, "seed": seed},
               CKPT_PATH)
    print(f"[1.1] SAVED -> {CKPT_PATH}")


if __name__ == "__main__":
    main()
