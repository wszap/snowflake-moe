# -*- coding: utf-8 -*-
"""任务1 终身学习（Training-Free Update）（版本 20001）

流程：
1.2 新领域数据：KJV Bible → 同一字符表 tokenize → 90/10 → data/new_domain_{train,test}.pt
1.3 加载 v3_base_298k.pt，冻结全部参数，每个 CellMoE 新增 8 个忆点，只解冻新忆点
1.4 新领域训练集训 1 epoch（lr=1e-3）→ checkpoints/v3_lifelong.pt
1.5 三重验证：莎士比亚测试集 PPL（遗忘）、新领域测试集 PPL（学会）、新忆点激活率（路由）
输出 output/results_snowflake_lifelong.csv
红线：不改架构、不调 lambda、每跑完一组 empty_cache + gc.collect
"""
import csv
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

CKPT_PATH = os.path.join(BASE, "checkpoints", "v3_base_298k.pt")
LIFELONG_CKPT = os.path.join(BASE, "checkpoints", "v3_lifelong.pt")
DATA_DIR = os.path.join(BASE, "data")
BIBLE_PATH = os.path.join(DATA_DIR, "bible_kjv.txt")
NEW_TRAIN = os.path.join(DATA_DIR, "new_domain_train.pt")
NEW_TEST = os.path.join(DATA_DIR, "new_domain_test.pt")
OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output",
                                       "results_snowflake_lifelong.csv"))

LAMBDA_ENT = 0.05
LAMBDA_MEM = 0.05
LAMBDA_ORG = 0.05
NEW_ADD = 8          # 每个细胞新增忆点数


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def encode_with_chars(text, chars):
    stoi = {c: i for i, c in enumerate(chars)}
    sp = stoi.get(' ', 0)
    return torch.tensor([stoi.get(c, sp) for c in text], dtype=torch.long)


def prepare_new_domain(chars):
    if os.path.exists(NEW_TRAIN) and os.path.exists(NEW_TEST):
        new_train = torch.load(NEW_TRAIN)
        new_test = torch.load(NEW_TEST)
        print(f"[1.2] 复用已存在数据: train={new_train.numel()} test={new_test.numel()}")
        return new_train, new_test
    with open(BIBLE_PATH, encoding='utf-8') as f:
        text = f.read()
    ids = encode_with_chars(text, chars)
    n = ids.numel()
    n_train = int(n * 0.9)
    new_train, new_test = ids[:n_train], ids[n_train:]
    os.makedirs(DATA_DIR, exist_ok=True)
    torch.save(new_train, NEW_TRAIN)
    torch.save(new_test, NEW_TEST)
    print(f"[1.2] KJV Bible tokenize: total={n} train={new_train.numel()} "
          f"test={new_test.numel()} 保存 -> {NEW_TRAIN} / {NEW_TEST}")
    return new_train, new_test


def add_new_memories(model, add=NEW_ADD, seed=2026):
    n = 0
    for layer in model.layers:
        for cell in layer.cells:
            cell.add_new_memory(add, seed=seed)
            n += 1
    print(f"[1.3] 扩展忆点: {n} 个细胞各 +{add}，共新增 {n * add} 个忆点")
    return n


def freeze_all_except_new(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in model.layers:
        for cell in layer.cells:
            cell.new_memory_keys.requires_grad_(True)
            cell.new_memory_assembly.requires_grad_(True)


def new_memory_utilization(model, ids, vocab_size, seq_len=64,
                           n_batch=20, batch_size=16, n_memory=24):
    """路由测试：新领域输入上，忆点检索 argmax 落在新忆点区间(>=n_memory)的比例。"""
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for _ in range(n_batch):
            xb, _ = lm_batch(ids, batch_size, seq_len)
            _, info = model(xb)
            for ci in info['cell_infos']:
                am = ci['memory_attn']                 # [N, n_memory+add]
                hits += int((am.argmax(-1) >= n_memory).sum().item())
                total += am.shape[0]
    model.train()
    return hits / max(1, total)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    set_seed(2026)
    vocab_size, train_ids, val_ids, chars = load_shakespeare()
    seq_len, batch_size = 64, 128

    # ---- 1.2 新领域数据 ----
    new_train, new_test = prepare_new_domain(chars)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 1.3 加载基线 + 冻结 + 加新忆点 ----
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    cfg = ckpt['cfg']
    model = HierarchicalCellMoE_LM(d=cfg['d'], vocab_size=cfg['vocab_size'],
                                   n_cells=cfg['n_cells'],
                                   n_organelles=cfg['n_organelles'],
                                   n_memory=cfg['n_memory'],
                                   topk_organelle=cfg['topk_organelle'],
                                   topk_cell=cfg['topk_cell'],
                                   L=cfg['L']).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    _, base_ppl = eval_ppl(model, val_ids, vocab_size, seq_len,
                           n_batch=20, batch_size=16)
    _, base_new_ppl = eval_ppl(model, new_test, vocab_size, seq_len,
                               n_batch=20, batch_size=16)
    print(f"[1.1] 基线: shakespeare_ppl={base_ppl:.3f} "
          f"(checkpoint记录={ckpt['final_ppl']:.3f})  new_domain_ppl={base_new_ppl:.3f}")
    add_new_memories(model)
    freeze_all_except_new(model)
    trainable = count_params(model)
    print(f"[1.3] 冻结完成，可训练参数={trainable}（仅新忆点）")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 1.4 只训练新忆点：新领域 1 epoch, lr=1e-3 ----
    t0 = time.time()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    n_steps = max(1, new_train.numel() // (seq_len * batch_size))
    model.train()
    loss_sum, n_st = 0.0, 0
    for i in range(n_steps):
        xb, yb = lm_batch(new_train, batch_size, seq_len, seed=2026 + i)
        opt.zero_grad(set_to_none=True)
        logits, info = model(xb)
        ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        w = torch.cat([ci['weights'] for ci in info['cell_infos']], dim=0)
        attn = torch.cat([ci['memory_attn'] for ci in info['cell_infos']], dim=0)
        entropy = -(w * torch.log(w + 1e-9)).sum(-1).mean()
        memory_usage = attn.mean(dim=0)
        aux_memory = (memory_usage * memory_usage).sum() * (cfg['n_memory'] + NEW_ADD)
        freq_soft = w.mean(dim=0)
        aux_organelle = (freq_soft * freq_soft).sum() * cfg['n_organelles']
        loss = (ce_loss - LAMBDA_ENT * entropy
                + LAMBDA_MEM * aux_memory + LAMBDA_ORG * aux_organelle)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        loss_sum += loss.item()
        n_st += 1
        if (i + 1) % 100 == 0:
            print(f"   step {i + 1}/{n_steps}  loss={loss_sum / n_st:.4f}  "
                  f"({(time.time() - t0):.0f}s)")
        del logits, loss
    print(f"[1.4] 新忆点训练完成: {n_steps} steps, avg_loss={loss_sum / n_st:.4f} "
          f"({(time.time() - t0):.0f}s)")
    os.makedirs(os.path.dirname(LIFELONG_CKPT), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                "base_shakespeare_ppl": base_ppl, "base_new_domain_ppl": base_new_ppl,
                "new_add": NEW_ADD}, LIFELONG_CKPT)
    print(f"[1.4] SAVED -> {LIFELONG_CKPT}")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 1.5 三重验证 ----
    _, life_ppl = eval_ppl(model, val_ids, vocab_size, seq_len,
                           n_batch=20, batch_size=16)
    _, life_new_ppl = eval_ppl(model, new_test, vocab_size, seq_len,
                               n_batch=20, batch_size=16)
    util = new_memory_utilization(model, new_test, vocab_size, seq_len,
                                  n_batch=20, batch_size=16, n_memory=cfg['n_memory'])
    print(f"[1.5] 三重验证: shakespeare_ppl={life_ppl:.3f} (基线{base_ppl:.3f})  "
          f"new_domain_ppl={life_new_ppl:.3f} (基线{base_new_ppl:.3f})  "
          f"new_memory_utilization={util:.3%}")

    # ---- 输出 CSV ----
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    new_file = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["phase", "shakespeare_ppl", "new_domain_ppl",
                        "new_memory_utilization"])
        w.writerow(["baseline", round(base_ppl, 4), round(base_new_ppl, 4), 0.0])
        w.writerow(["lifelong_1epoch", round(life_ppl, 4), round(life_new_ppl, 4),
                    round(util, 4)])
    print(f"[1.5] CSV -> {OUT_CSV}")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
