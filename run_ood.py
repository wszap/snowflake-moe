# -*- coding: utf-8 -*-
"""任务2 OOD 泛化（版本 20001）

流程：
2.1 莎士比亚按"句长 + 标点密度"切分喜剧/悲剧各 50%（无类型标注，按用户指定启发式）
2.2 喜剧子集 90/10 -> comedy_train / comedy_val(ID)，悲剧 90/10 -> tragedy_val(OOD)
2.3 v3 单级 CellMoE（SnowflakeMoE_LM 203K, seed2026）喜剧训 5ep，测 ID/OOD
2.4 fixed MoE（MarvisMoE_LM 259K, seed2026）同配置喜剧训 5ep，测 ID/OOD
2.5 记 OOD/ID 比值对比 -> output/results_snowflake_ood.csv
红线：不改架构、不调 lambda、GPU<80°C（超则暂停 10 分钟）、每模型跑完 empty_cache+gc
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

from snowflake_moe import SnowflakeMoE_LM  # noqa: E402
from train_lm import MarvisMoE_LM, load_shakespeare, lm_batch, eval_ppl  # noqa: E402
from marvis_moe import Config  # noqa: E402

DATA_PATH = os.path.join(BASE, "data", "tinyshakespeare", "input.txt")
OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output",
                                       "results_snowflake_ood.csv"))

FIXED_CFG = dict(d=64, h=128, E=4, S=1, L=2, topk=2)
SEQ_LEN, BATCH_SIZE, EPOCHS, LR, SEED = 64, 128, 5, 3e-4, 2026

# CellMoE 机制正则（沿用 00003 防坍缩机制，不调 lambda）
LAMBDA_ENT = 0.05
LAMBDA_MEM = 0.05
LAMBDA_ORG = 0.05

PUNCT = set(".,;:!?\"'()[]-")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def check_gpu_temp(max_c=79, wait_s=600):
    """红线：GPU < 80°C，超则暂停 10 分钟。"""
    if DEVICE != "cuda":
        return
    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"], text=True)
            temp = int(out.strip().splitlines()[0])
            if temp < max_c:
                return
            print(f"[TEMP] GPU {temp}°C >= {max_c}°C，暂停 {wait_s}s")
            time.sleep(wait_s)
        except Exception as e:  # noqa: BLE001
            print(f"[TEMP] 查询失败，继续: {e}")
            return


def split_comedy_tragedy(text):
    """按句长 + 标点密度启发式切分喜剧/悲剧各 50%。

    无类型标注，按用户指定：每句记句长与标点密度，归一化求和得
    score = z(句长) + z(标点密度)，score 高 50% 判为喜剧、低 50% 判为悲剧，
    组内保持原文本顺序拼接。
    """
    import re
    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p for p in parts if p.strip()]
    lens = np.array([len(p) for p in parts], dtype=float)
    dens = np.array([sum(1 for ch in p if ch in PUNCT) / max(1, len(p))
                     for p in parts], dtype=float)
    zl = (lens - lens.mean()) / (lens.std() + 1e-9)
    zd = (dens - dens.mean()) / (dens.std() + 1e-9)
    score = zl + zd
    order = np.argsort(score)          # 升序
    n = len(order)
    half = n // 2
    comedy_idx = set(order[half:].tolist())   # score 高 50% -> 喜剧
    tragedy_idx = set(order[:half].tolist())  # score 低 50% -> 悲剧
    comedy = "".join(parts[i] for i in range(n) if i in comedy_idx)
    tragedy = "".join(parts[i] for i in range(n) if i in tragedy_idx)
    print(f"[2.1] 句子总数={n}  喜剧字符={len(comedy)}  悲剧字符={len(tragedy)}")
    return comedy, tragedy


def encode_with_chars(text, chars):
    stoi = {c: i for i, c in enumerate(chars)}
    sp = stoi.get(' ', 0)
    return torch.tensor([stoi.get(c, sp) for c in text], dtype=torch.long)


def train_model(model, train_ids, vocab_size, tag):
    """喜剧子集训 5ep（warmup 5%，lr=3e-4）。返回 (train_loss, sec)。"""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_steps = max(1, train_ids.numel() // (SEQ_LEN * BATCH_SIZE))
    total_steps = n_steps * EPOCHS
    warmup = max(1, int(total_steps * 0.05))

    def lr_at(step):
        return LR * (step + 1) / warmup if step < warmup else LR

    t0 = time.time()
    step_global = 0
    for epoch in range(1, EPOCHS + 1):
        check_gpu_temp()
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
            if isinstance(info, dict):      # CellMoE：沿用 00003 机制正则
                w = info['weights']
                entropy = -(w * torch.log(w + 1e-9)).sum(-1).mean()
                memory_usage = info['memory_attn'].mean(dim=0)
                aux_memory = (memory_usage * memory_usage).sum() * info['memory_attn'].shape[-1]
                freq_soft = w.mean(dim=0)
                aux_organelle = (freq_soft * freq_soft).sum() * w.shape[-1]
                loss = (ce_loss - LAMBDA_ENT * entropy
                        + LAMBDA_MEM * aux_memory + LAMBDA_ORG * aux_organelle)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step_global += 1
            e_sum += loss.item()
            e_n += 1
            del logits, loss
        print(f"[{tag} EPOCH {epoch}/{EPOCHS}] train_loss={e_sum / e_n:.4f} "
              f"({(time.time() - t0):.0f}s)")
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return e_sum / e_n, time.time() - t0


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    set_seed(SEED)
    vocab_size, _, _, chars = load_shakespeare(DATA_PATH)

    # ---- 2.1 喜剧/悲剧切分（一次性，两模型共享） ----
    comedy_text, tragedy_text = split_comedy_tragedy(open(DATA_PATH, encoding='utf-8').read())
    comedy_ids = encode_with_chars(comedy_text, chars)
    tragedy_ids = encode_with_chars(tragedy_text, chars)
    n_c = comedy_ids.numel()
    n_t = tragedy_ids.numel()
    comedy_train, comedy_val = comedy_ids[:int(n_c * 0.9)], comedy_ids[int(n_c * 0.9):]
    tragedy_train, tragedy_val = tragedy_ids[:int(n_t * 0.9)], tragedy_ids[int(n_t * 0.9):]
    print(f"[2.2] comedy_train={comedy_train.numel()} comedy_val(ID)={comedy_val.numel()} "
          f"tragedy_train={tragedy_train.numel()} tragedy_val(OOD)={tragedy_val.numel()}")

    rows = []
    # ---- 2.3 v3 单级 CellMoE（203K） ----
    cell = SnowflakeMoE_LM(d=64, vocab_size=vocab_size, n_organelles=20,
                           n_memory=64, topk_organelle=4, L=2).to(DEVICE)
    print(f"[2.3] CellMoE params={count_params(cell)}")
    _, sec = train_model(cell, comedy_train, vocab_size, "cellmoe")
    _, id_ppl = eval_ppl(cell, comedy_val, vocab_size, SEQ_LEN, n_batch=20, batch_size=16)
    _, ood_ppl = eval_ppl(cell, tragedy_val, vocab_size, SEQ_LEN, n_batch=20, batch_size=16)
    ratio = ood_ppl / id_ppl
    rows.append(("cellmoe_v3", round(id_ppl, 4), round(ood_ppl, 4), round(ratio, 4)))
    print(f"[2.3] CellMoE  ID(喜剧)={id_ppl:.3f}  OOD(悲剧)={ood_ppl:.3f}  "
          f"OOD/ID={ratio:.4f}  ({sec:.0f}s)")
    del cell
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 2.4 fixed MoE（259K） ----
    fixed = MarvisMoE_LM(Config(**FIXED_CFG), vocab_size).to(DEVICE)
    print(f"[2.4] fixed MoE params={count_params(fixed)}")
    _, sec = train_model(fixed, comedy_train, vocab_size, "fixed")
    _, id_ppl = eval_ppl(fixed, comedy_val, vocab_size, SEQ_LEN, n_batch=20, batch_size=16)
    _, ood_ppl = eval_ppl(fixed, tragedy_val, vocab_size, SEQ_LEN, n_batch=20, batch_size=16)
    ratio = ood_ppl / id_ppl
    rows.append(("fixed_moe", round(id_ppl, 4), round(ood_ppl, 4), round(ratio, 4)))
    print(f"[2.4] fixed  ID(喜剧)={id_ppl:.3f}  OOD(悲剧)={ood_ppl:.3f}  "
          f"OOD/ID={ratio:.4f}  ({sec:.0f}s)")
    del fixed
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 2.5 输出 CSV ----
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "id_ppl", "ood_ppl", "ood_id_ratio"])
        w.writerows(rows)
    print(f"[2.5] CSV -> {OUT_CSV}")
    print("[DONE] 任务2 完成")


if __name__ == "__main__":
    main()
