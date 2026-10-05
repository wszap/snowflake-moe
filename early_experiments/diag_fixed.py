# -*- coding: utf-8 -*-
"""fixed ID=37.44 异常诊断：多窗口 eval 定位"""
import math, os, sys, random, time
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from train_lm import MarvisMoE_LM, lm_batch, load_shakespeare  # noqa: E402
from marvis_moe import Config  # noqa: E402

SEED = 2026
SEQ_LEN, BATCH_SIZE, EPOCHS, LR = 64, 128, 5, 3e-4
COMEDY = os.path.join(BASE, "data_shk_comedy.txt")
HISTORY = os.path.join(BASE, "data_shk_history.txt")

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
            logits, _ = model(xb)
            loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sg += 1; s += loss.item(); n += 1
        print(f"[{tag} ep{ep}] {s/n:.4f} ({(time.time()-t0):.0f}s)")
    return s / n

def eval_ppl_detail(model, ids, vocab_size, n_batch=20, batch_size=16, tag=""):
    model.eval()
    ces = []
    with torch.no_grad():
        for _ in range(n_batch):
            xb, yb = lm_batch(ids, batch_size, SEQ_LEN)
            logits, _ = model(xb)
            ces.append(F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1)).item())
    model.train()
    ces = np.array(ces)
    print(f"[{tag}] n={len(ces)} mean={ces.mean():.4f} ppl={math.exp(ces.mean()):.3f} "
          f"min={ces.min():.4f} max={ces.max():.4f} 高值(>3.2)={np.sum(ces>3.2)}")
    return ces.mean()

def main():
    set_seed(SEED)
    ct = open(COMEDY, encoding='utf-8').read()
    ht = open(HISTORY, encoding='utf-8').read()
    stoi = build_vocab(ct, ht)
    vocab_size = len(stoi)
    cids = encode(ct, stoi); hids = encode(ht, stoi)
    n_c = cids.numel()
    c_train, c_val = cids[:int(n_c*0.9)], cids[int(n_c*0.9):]
    fixed = MarvisMoE_LM(Config(d=64, h=128, E=4, S=1, L=2, topk=2), vocab_size).to(DEVICE)
    print("fixed params", sum(p.numel() for p in fixed.parameters() if p.requires_grad))
    train(fixed, c_train, vocab_size, "fixed")
    # 窗口诊断
    eval_ppl_detail(fixed, c_train, vocab_size, tag="c_train 随机")
    eval_ppl_detail(fixed, c_val, vocab_size, tag="c_val 随机")
    eval_ppl_detail(fixed, hids, vocab_size, tag="history 随机")
    # val 分 5 段连续评估
    seg = c_val.numel() // 5
    for i in range(5):
        eval_ppl_detail(fixed, c_val[i*seg:(i+1)*seg], vocab_size, n_batch=10, tag=f"c_val seg{i}")

if __name__ == "__main__":
    main()
