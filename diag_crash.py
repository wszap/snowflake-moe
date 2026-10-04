# -*- coding: utf-8 -*-
"""定位 eval_full 崩溃：训练 cell 5ep 后逐段 eval，faulthandler 抓 C++ 崩溃"""
import faulthandler, os, sys, math, random, time
import numpy as np
import torch
import torch.nn.functional as F

faulthandler.enable()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from snowflake_moe import SnowflakeMoE_LM  # noqa: E402
from train_lm import lm_batch  # noqa: E402

SEED = 2026
SEQ_LEN, BATCH_SIZE, EPOCHS, LR = 64, 128, 5, 3e-4
COMEDY = os.path.join(BASE, "data_shk_comedy.txt")
HISTORY = os.path.join(BASE, "data_shk_history.txt")
LAMBDA_ENT = LAMBDA_MEM = LAMBDA_ORG = 0.05

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE == "cuda": torch.cuda.manual_seed_all(seed)

def build_vocab(*texts):
    chars = sorted(set("".join(texts)))
    return {c: i for i, c in enumerate(chars)}

def encode(text, stoi):
    sp = stoi.get(' ', 0)
    return torch.tensor([stoi.get(c, sp) for c in text], dtype=torch.long)

def train(model, train_ids, vocab_size, tag):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_steps = max(1, train_ids.numel() // (SEQ_LEN * BATCH_SIZE))
    total = n_steps * EPOCHS
    warmup = max(1, int(total * 0.05))
    def lr_at(s): return LR * (s + 1) / warmup if s < warmup else LR
    t0 = time.time(); sg = 0
    for ep in range(1, EPOCHS + 1):
        model.train(); s = 0.0; n = 0
        for _ in range(n_steps):
            xb, yb = lm_batch(train_ids, BATCH_SIZE, SEQ_LEN, seed=SEED + sg)
            for g in opt.param_groups: g['lr'] = lr_at(sg)
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
                loss = (ce_loss - LAMBDA_ENT * entropy + LAMBDA_MEM * aux_memory
                        + LAMBDA_ORG * aux_organelle)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sg += 1; s += loss.item(); n += 1
        print(f"[{tag} ep{ep}] {s/n:.4f} ({(time.time()-t0):.0f}s)", flush=True)
    return s / n

def eval_seg(model, ids, vocab_size, start, end, batch_size=16, tag=""):
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
            total_ce += ce * k; total_n += k; s += k
    model.train()
    print(f"[{tag}] seg[{start}:{end}] n={total_n} mean={total_ce/max(total_n,1):.4f}", flush=True)
    return total_ce / max(total_n, 1)

def main():
    set_seed(SEED)
    ct = open(COMEDY, encoding='utf-8').read()
    ht = open(HISTORY, encoding='utf-8').read()
    stoi = build_vocab(ct, ht)
    vocab_size = len(stoi)
    cids = encode(ct, stoi); hids = encode(ht, stoi)
    n_c = cids.numel()
    c_train, c_val = cids[:int(n_c*0.9)], cids[int(n_c*0.9):]
    print("vocab", vocab_size, "c_train", c_train.numel(), "c_val", c_val.numel(),
          "hids", hids.numel(), flush=True)
    cell = SnowflakeMoE_LM(d=64, vocab_size=vocab_size, n_organelles=20,
                           n_memory=64, topk_organelle=4, L=2).to(DEVICE)
    print("params", sum(p.numel() for p in cell.parameters() if p.requires_grad), flush=True)
    train(cell, c_train, vocab_size, "cell")
    print("training done, eval c_val...", flush=True)
    eval_seg(cell, c_val, vocab_size, 0, c_val.numel(), tag="c_val")
    print("c_val ok, eval hids...", flush=True)
    eval_seg(cell, hids, vocab_size, 0, hids.numel(), tag="hids")
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
