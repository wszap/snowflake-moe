# -*- coding: utf-8 -*-
"""batch=128 1-epoch GPU vs CPU 测速（用户指令第二步）"""
import sys, os, time, math, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from train_lm import load_shakespeare, Config, train_lm, DATA_PATH

def main():
    vocab_size, train_ids, val_ids, chars = load_shakespeare(DATA_PATH)
    cfg = Config(d=64, h=128, E=4, S=1, L=2, topk=2, out_dim=vocab_size)
    results = {}
    for device in ("cuda", "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA unavailable, skip")
            continue
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        t0 = time.time()
        model, h, _ = train_lm(cfg, vocab_size, train_ids, val_ids,
                               council_mode="fixed", seed=2026, epochs=1,
                               batch_size=128, seq_len=64, lr=3e-4, accum=1,
                               n_val_batch=20, device=device, verbose=False,
                               use_clip=False, use_warmup=False)
        dt = time.time() - t0
        results[device] = (dt, h["val_ppl_final"])
        print(f"[{device}] 1 epoch: {dt:.2f}s  val_ppl={h['val_ppl_final']:.4f}  peak_mem={h.get('peak_mem_gb','n/a')}")
        if device == "cuda":
            torch.cuda.empty_cache()
    if len(results) == 2:
        speedup = results["cpu"][0] / results["cuda"][0]
        print(f"SPEEDUP batch=128: cpu {results['cpu'][0]:.2f}s / gpu {results['cuda'][0]:.2f}s = {speedup:.2f}x")

if __name__ == "__main__":
    main()
