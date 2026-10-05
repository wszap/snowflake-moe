# -*- coding: utf-8 -*-
"""最小复现 eval_full 崩溃"""
import os, sys, math
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from snowflake_moe import SnowflakeMoE_LM  # noqa: E402

SEQ_LEN = 64

def eval_full(model, ids, vocab_size, batch_size=16):
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
            if s % 10000 == 0:
                print(f"  ... s={s} n={n}", flush=True)
    model.train()
    return total_ce / total_n, math.exp(total_ce / total_n)

print("device", DEVICE, flush=True)
ids = torch.randint(0, 90, (100000,), dtype=torch.long)
model = SnowflakeMoE_LM(d=64, vocab_size=90, n_organelles=20, n_memory=64,
                        topk_organelle=4, L=2).to(DEVICE)
print("model ok", flush=True)
ce, ppl = eval_full(model, ids, 90)
print("eval ok", ce, ppl, flush=True)
