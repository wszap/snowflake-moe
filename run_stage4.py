# -*- coding: utf-8 -*-
"""阶段四：免疫系统对比（A 开免疫+修剪 vs B 关免疫+修剪），莎士比亚 10 epoch
- 加载 improved_v3_2026.pt，深拷贝 A/B 同起点
- A：每 epoch 免疫 check+treat；每 2 epoch prune_memories
- B：不调用免疫；每 2 epoch prune_memories
- 超参同 stage1（lr 3e-4, batch 128, seq 64, warmup 5%, clip 1.0, seed 2026）
- 输出 output/results_immune_improved.csv
- 验收：A 利用率 >= B*1.1 且 A PPL <= B PPL
"""
import csv
import copy
import gc
import math
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

from snowflake_moe_improved import ImprovedHierarchicalCellMoE_LM, prune_memories  # noqa: E402
from train_lm import load_shakespeare, lm_batch  # noqa: E402

SEQ_LEN, BATCH_SIZE, EPOCHS, LR, SEED = 64, 128, 10, 3e-4, 2026
LAMBDA_ENT, LAMBDA_MEM, LAMBDA_ORG = 0.05, 0.05, 0.05
N_ORG = 6
CKPT = os.path.join(BASE, "checkpoints", "improved_v3_2026.pt")
OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output",
                                       "results_immune_improved.csv"))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gpu_temp():
    if DEVICE != "cuda":
        return None
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def build_model(vocab_size, ckpt):
    model = ImprovedHierarchicalCellMoE_LM(
        d=64, vocab_size=vocab_size, n_cells=4, n_organelles=N_ORG, n_memory=24,
        topk_organelle=4, topk_cell=2, L=2, gate_ent_reg=True).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    return model


def eval_full_info(model, ids, vocab_size, batch_size=16):
    """确定性全量评估：返回 (avg_ce, ppl, freq[6], util_ent)。"""
    model.eval()
    k = batch_size * SEQ_LEN
    n = ids.numel()
    total_ce, total_n = 0.0, 0
    freq = torch.zeros(N_ORG, device=DEVICE)
    with torch.no_grad():
        s = 0
        while s + k + 1 <= n:
            x = ids[s:s + k].view(batch_size, SEQ_LEN).to(DEVICE)
            y = ids[s + 1:s + k + 1].view(batch_size, SEQ_LEN).to(DEVICE)
            logits, info = model(x)
            ce = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
            total_ce += ce * k
            total_n += k
            if isinstance(info, dict) and info.get("cell_infos"):
                for ci in info["cell_infos"]:
                    t = ci["topk_idx"].reshape(-1)
                    freq.scatter_add_(0, t, torch.ones(t.numel(), device=DEVICE))
            s += k
    model.train()
    avg_ce = total_ce / total_n
    p = freq / freq.sum().clamp_min(1.0)
    util_ent = float(-(p * torch.log(p + 1e-9)).sum().item())
    return avg_ce, math.exp(avg_ce), freq.cpu(), util_ent


def immune_round(model, x_sample):
    """对每层每 cell 免疫体检+治疗。x_sample: [B, d] 输入。"""
    replaced = 0
    for layer in model.layers:
        for cell in layer.cells:
            rep = cell.immunity.check(cell, x_sample)
            replaced += cell.immunity.treat(cell, rep)
    return replaced


def prune_round(model):
    total = 0
    for layer in model.layers:
        for cell in layer.cells:
            total += prune_memories(cell, min_usage=0.05)
    return total


def train_one(model, train_ids, val_ids, vocab_size, immune, tag):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_steps = max(1, train_ids.numel() // (SEQ_LEN * BATCH_SIZE))
    total_steps = n_steps * EPOCHS
    warmup = max(1, int(total_steps * 0.05))

    def lr_at(st):
        return LR * (st + 1) / warmup if st < warmup else LR

    t0 = time.time()
    step_global = 0
    # 免疫用固定样本：训练集一批 token 的 layer 输入
    xb0, _ = lm_batch(train_ids, BATCH_SIZE, SEQ_LEN, seed=999)
    with torch.no_grad():
        xv0 = model.in_proj(model.embed(xb0.to(DEVICE))).reshape(-1, model.d)
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
            if isinstance(info, dict):
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
            del logits, loss
        print(f"[S4 {tag} EPOCH {epoch}/{EPOCHS}] train_loss={e_sum / e_n:.4f} "
              f"({time.time() - t0:.0f}s) gpu={gpu_temp()}C", flush=True)
        if immune:
            r = immune_round(model, xv0)
            print(f"[S4 {tag}] immune treat replaced={r}", flush=True)
        if epoch % 2 == 0:
            pr = prune_round(model)
            print(f"[S4 {tag}] prune removed={pr} mem_now={[c.memory_keys.shape[0] for l in model.layers for c in l.cells]}",
                  flush=True)
            opt = torch.optim.Adam(model.parameters(), lr=LR)  # 修剪后重建
            for g in opt.param_groups:
                g['lr'] = lr_at(step_global)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    val_ce, val_ppl, freq, util_ent = eval_full_info(model, val_ids, vocab_size)
    active = int((freq > 0).sum().item())
    print(f"[S4 {tag}] FINAL val_ppl={val_ppl:.4f} util_ent={util_ent:.4f} "
          f"active_org={active}/{N_ORG} freq={freq.tolist()}")
    return dict(tag=tag, val_ppl=val_ppl, util_ent=util_ent,
                active=active, freq=freq.tolist(), sec=round(time.time() - t0, 1))


def main():
    set_seed(SEED)
    vocab_size, train_ids, val_ids, chars = load_shakespeare()
    print(f"[S4] vocab={vocab_size} train={train_ids.numel()} val={val_ids.numel()} gpu={gpu_temp()}C")
    ckpt = torch.load(CKPT, map_location=DEVICE)
    ma = build_model(vocab_size, ckpt)
    mb = copy.deepcopy(ma).to(DEVICE)
    # 加强免疫强度：替换 25% 细胞器，噪声 0.05（默认 0.1/0.01 太温和）
    for layer in ma.layers:
        for cell in layer.cells:
            cell.immunity.replace_frac = 0.25
            cell.immunity.noise = 0.05
    print(f"[S4] A/B 同起点 params={count_params(ma)} (A 免疫强度 replace=0.25 noise=0.05)")

    ra = train_one(ma, train_ids, val_ids, vocab_size, True, "A")
    rb = train_one(mb, train_ids, val_ids, vocab_size, False, "B")

    # 利用率标准化：exp(ent)/N_ORG
    util_a = math.exp(ra["util_ent"]) / N_ORG
    util_b = math.exp(rb["util_ent"]) / N_ORG

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "immune", "val_ppl", "util_ent", "util_norm",
                    "active_org", "freq", "sec"])
        w.writerow(["A", 1, ra["val_ppl"], ra["util_ent"], round(util_a, 4),
                    ra["active"], ra["freq"], ra["sec"]])
        w.writerow(["B", 0, rb["val_ppl"], rb["util_ent"], round(util_b, 4),
                    rb["active"], rb["freq"], rb["sec"]])
    print(f"[S4] CSV -> {OUT_CSV}")

    # ---- checkpoint：最终模型保存（不改训练逻辑）----
    for tag, model, note in (("stage4_immune", ma, "免疫组A"),
                             ("stage4_control", mb, "对照组B")):
        ckpt_path = os.path.join(BASE, "checkpoints", f"{tag}_{SEED}.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"model_state": model.state_dict(),
                    "config": dict(d=128, vocab_size=vocab_size, n_cells=4,
                                   n_organelles=8, n_memory=32,
                                   topk_organelle=4, topk_cell=2, L=4,
                                   seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                                   epochs=EPOCHS, seed=SEED),
                    "group": note}, ckpt_path)
        print(f"[S4] CKPT saved -> {ckpt_path}", flush=True)

    # 验收
    ppl_ok = ra["val_ppl"] <= rb["val_ppl"]
    util_ok = util_a >= util_b * 1.1
    print(f"[S4] util_norm A={util_a:.4f} B={util_b:.4f} "
          f"(A/B={util_a / util_b:.3f}, 要求>=1.10)")
    print(f"[S4] val_ppl A={ra['val_ppl']:.4f} B={rb['val_ppl']:.4f} "
          f"(要求 A<=B)")
    if ppl_ok and util_ok:
        print("[S4 PASS] 免疫组利用率提升 >=10% 且 PPL 不劣化")
    else:
        print(f"[S4 FAIL] ppl_ok={ppl_ok} util_ok={util_ok}")


if __name__ == "__main__":
    main()
