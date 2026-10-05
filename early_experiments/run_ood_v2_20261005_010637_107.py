# -*- coding: utf-8 -*-
"""任务2 OOD 泛化 v2（主题切分版，确定性全量评估）

2.1 直接读取 PG100 主题切分文本（喜剧 vs 历史剧，难度差 11.2% 已接受）
2.2 comedy 90/10 -> comedy_train / comedy_val(ID)；history 全部 -> OOD 测试集
2.3 v3 单级 CellMoE（211K, seed2026）喜剧训 5ep，测 ID/OOD（全量）
2.4 fixed MoE（263K, seed2026）同配置喜剧训 5ep，测 ID/OOD（全量）
2.5 记 OOD/ID 比值对比 -> output/results_ood_v2.csv
红线：不改架构、不调 lambda、GPU<80°C、每模型跑完 empty_cache+gc
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
from train_lm import MarvisMoE_LM, lm_batch  # noqa: E402
from marvis_moe import Config  # noqa: E402

COMEDY_PATH = os.path.join(BASE, "data_shk_comedy.txt")
HISTORY_PATH = os.path.join(BASE, "data_shk_history.txt")
OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output",
                                       "results_ood_v2.csv"))

FIXED_CFG = dict(d=64, h=128, E=4, S=1, L=2, topk=2)
SEQ_LEN, BATCH_SIZE, EPOCHS, LR, SEED = 64, 128, 5, 3e-4, 2026

LAMBDA_ENT = 0.05
LAMBDA_MEM = 0.05
LAMBDA_ORG = 0.05


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def check_gpu_temp(max_c=79, wait_s=600):
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
            print(f"[TEMP] GPU {temp}°C >= {max_c}°C，暂停 {wait_s}s", flush=True)
            time.sleep(wait_s)
        except Exception:  # noqa: BLE001
            return


def build_vocab(*texts):
    chars = sorted(set("".join(texts)))
    return {c: i for i, c in enumerate(chars)}


def encode(text, stoi):
    sp = stoi.get(' ', 0)
    return torch.tensor([stoi.get(c, sp) for c in text], dtype=torch.long)


def train_model(model, train_ids, vocab_size, tag):
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
            if isinstance(info, dict):
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
        print(f"[{tag} EPOCH {epoch}/{EPOCHS}] train_loss={e_sum / e_n:.4f} "
              f"({(time.time() - t0):.0f}s)", flush=True)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return e_sum / e_n, time.time() - t0


def eval_seg(model, ids, vocab_size, start, end, batch_size=16, tag=""):
    """确定性全量评估：连续分块覆盖 [start, end)，返回 (avg_ce, ppl)。"""
    k = batch_size * SEQ_LEN
    total_ce, total_n = 0.0, 0
    model.eval()
    with torch.no_grad():
        s = start
        while s + k + 1 <= end:
            x = ids[s:s + k].view(batch_size, SEQ_LEN).to(DEVICE)
            y = ids[s + 1:s + k + 1].view(batch_size, SEQ_LEN).to(DEVICE)
            logits, _ = model(x)
            ce = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
            total_ce += ce * k
            total_n += k
            s += k
    model.train()
    avg_ce = total_ce / max(total_n, 1)
    print(f"[{tag}] n={total_n} ce={avg_ce:.4f} ppl={math.exp(avg_ce):.4f}", flush=True)
    return avg_ce, math.exp(avg_ce)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    set_seed(SEED)
    comedy_text = open(COMEDY_PATH, encoding='utf-8').read()
    history_text = open(HISTORY_PATH, encoding='utf-8').read()
    print(f"[2.1] comedy={len(comedy_text)} chars  history={len(history_text)} chars",
          flush=True)

    stoi = build_vocab(comedy_text, history_text)
    vocab_size = len(stoi)
    comedy_ids = encode(comedy_text, stoi)
    history_ids = encode(history_text, stoi)

    n_c = comedy_ids.numel()
    comedy_train = comedy_ids[:int(n_c * 0.9)]
    comedy_val = comedy_ids[int(n_c * 0.9):]
    print(f"[2.2] vocab={vocab_size}  comedy_train={comedy_train.numel()} "
          f"comedy_val(ID)={comedy_val.numel()}  history_val(OOD)={history_ids.numel()}",
          flush=True)

    rows = []
    # ---- 2.3 v3 单级 CellMoE（211K） ----
    cell = SnowflakeMoE_LM(d=64, vocab_size=vocab_size, n_organelles=20,
                           n_memory=64, topk_organelle=4, L=2).to(DEVICE)
    print(f"[2.3] CellMoE params={count_params(cell)}", flush=True)
    _, sec = train_model(cell, comedy_train, vocab_size, "cellmoe")
    _, id_ppl = eval_seg(cell, comedy_val, vocab_size, 0, comedy_val.numel(), tag="cellmoe ID")
    _, ood_ppl = eval_seg(cell, history_ids, vocab_size, 0, history_ids.numel(), tag="cellmoe OOD")
    ratio = ood_ppl / id_ppl
    rows.append(("cellmoe_v3", round(id_ppl, 4), round(ood_ppl, 4), round(ratio, 4)))
    print(f"[2.3] CellMoE  ID(喜剧)={id_ppl:.3f}  OOD(历史)={ood_ppl:.3f}  "
          f"OOD/ID={ratio:.4f}  ({sec:.0f}s)", flush=True)
    del cell
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 2.4 fixed MoE（263K） ----
    fixed = MarvisMoE_LM(Config(**FIXED_CFG), vocab_size).to(DEVICE)
    print(f"[2.4] fixed MoE params={count_params(fixed)}", flush=True)
    _, sec = train_model(fixed, comedy_train, vocab_size, "fixed")
    _, id_ppl = eval_seg(fixed, comedy_val, vocab_size, 0, comedy_val.numel(), tag="fixed ID")
    _, ood_ppl = eval_seg(fixed, history_ids, vocab_size, 0, history_ids.numel(), tag="fixed OOD")
    ratio = ood_ppl / id_ppl
    rows.append(("fixed_moe", round(id_ppl, 4), round(ood_ppl, 4), round(ratio, 4)))
    print(f"[2.4] fixed  ID(喜剧)={id_ppl:.3f}  OOD(历史)={ood_ppl:.3f}  "
          f"OOD/ID={ratio:.4f}  ({sec:.0f}s)", flush=True)
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
    print(f"[2.5] CSV -> {OUT_CSV}", flush=True)
    print("[DONE] 任务2 v2 完成", flush=True)


if __name__ == "__main__":
    main()
