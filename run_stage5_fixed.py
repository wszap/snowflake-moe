# -*- coding: utf-8 -*-
"""阶段五 Fixed 对照：dense FFN transformer（与 CellMoE 同配置对齐）
- 同数据/同超参/同 seed：25MB TinyStories, seq=128, batch=64, epochs=5,
  lr=3e-4, warmup 5%, clip 1.0, seed 2026
- width(E) 调至 14 使参数对齐 CellMoE 1.82M（差距<10%）
- 输出 output/results_tinystories_fixed.csv；每 50 step 打印 data/forward 诊断
  与 GPU/CPU 温度（红线 GPU<80C / CPU<85C）
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

SEQ_LEN, BATCH_SIZE, EPOCHS, LR, SEED = 128, 64, 5, 3e-4, 2026
WIDTH_E = 14          # FFN 宽度因子：参数对齐 CellMoE 1.82M（E=8 仅 1.1M 差 40%）
DIAG_EVERY = 50
GPU_TEMP_MAX = 80
CPU_TEMP_MAX = 85
MAX_TOTAL_SEC = 55 * 60
EPOCH_BUDGET_SEC = 540
DATA = os.path.join(BASE, "tinystories_100mb.txt")
OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output",
                                       "results_tinystories_fixed.csv"))


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


def load_tinystories(path=DATA, val_frac=0.1, seed=2026, use_mb=25):
    with open(path, "rb") as f:
        raw = f.read()
    if use_mb:
        raw = raw[:use_mb * 1024 * 1024]
    text = raw.decode("utf-8", errors="replace")
    chars = sorted(set(text))
    vocab_size = len(chars)
    if all(ord(c) < 256 for c in chars):
        lut = np.zeros(256, dtype=np.int64)
        lut[:] = -1
        for i, c in enumerate(chars):
            lut[ord(c)] = i
        arr = lut[np.frombuffer(raw, dtype=np.uint8)]
        if (arr < 0).any():
            raise ValueError("non-ASCII bytes found, fallback needed")
        data = torch.from_numpy(arr.astype(np.int64))
    else:
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
    offsets = idx.unsqueeze(1) + torch.arange(seq_len)
    xb = ids[offsets]
    yb = ids[offsets + 1]
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


def gpu_temp():
    if DEVICE != "cuda":
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def cpu_temp():
    """尽力读取 CPU 温度（WMI 热区），不可用返回 None。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance -Namespace root/wmi -ClassName "
             "MSAcpi_ThermalZoneTemperature).CurrentTemperature"],
            capture_output=True, text=True, timeout=8)
        t = out.stdout.strip().splitlines()
        if not t:
            return None
        # 返回值为开尔文*10
        return int(t[0]) / 10.0 - 273.15
    except Exception:
        return None


def cpu_load():
    """读取 CPU 总负载百分比，不可用返回 None。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Counter '\\Processor(_Total)\\% Processor Time' "
             "-SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue"],
            capture_output=True, text=True, timeout=8)
        v = float(out.stdout.strip().splitlines()[-1])
        return v
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
            logits = model(x)
            ce = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
            total_ce += ce * k
            total_n += k
            s += k
    model.train()
    avg_ce = total_ce / total_n
    return avg_ce, math.exp(avg_ce)


def train_one(model, train_ids, val_ids, vocab_size, tag):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n_steps = max(1, train_ids.numel() // (SEQ_LEN * BATCH_SIZE))
    epochs = EPOCHS
    total_steps = n_steps * epochs
    warmup = max(1, int(total_steps * 0.05))

    def lr_at(st):
        return LR * (st + 1) / warmup if st < warmup else LR

    t_start = time.time()
    step_global = 0
    util_sum, util_n = 0.0, 0
    w_data, w_fwd, w_opt, w_n = 0.0, 0.0, 0.0, 0
    t_data_all, t_fwd_all, t_opt_all = 0.0, 0.0, 0.0
    epoch_secs = []
    prefetch = None
    ct_hist = []

    for epoch in range(1, epochs + 1):
        model.train()
        e_sum, e_n = 0.0, 0
        e_t0 = time.time()
        for _ in range(n_steps):
            t0 = time.time()
            if prefetch is None:
                xb, yb = lm_batch(train_ids, BATCH_SIZE, SEQ_LEN,
                                  seed=SEED + step_global)
            else:
                xb, yb = prefetch
            prefetch = lm_batch(train_ids, BATCH_SIZE, SEQ_LEN,
                                seed=SEED + step_global + 1)
            t_data = time.time() - t0

            for g in opt.param_groups:
                g['lr'] = lr_at(step_global)
            opt.zero_grad(set_to_none=True)

            t0 = time.time()
            logits = model(xb)
            t_fwd = time.time() - t0

            t0 = time.time()
            loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_opt = time.time() - t0

            step_global += 1
            e_sum += loss.item()
            e_n += 1
            w_data += t_data
            w_fwd += t_fwd
            w_opt += t_opt
            w_n += 1
            t_data_all += t_data
            t_fwd_all += t_fwd
            t_opt_all += t_opt

            if step_global % DIAG_EVERY == 0:
                u = gpu_util()
                gt = gpu_temp()
                ct = cpu_temp()
                if u is not None:
                    util_sum += u
                    util_n += 1
                ct_note = ""
                if ct is not None:
                    ct_hist.append(ct)
                    if (len(ct_hist) >= 3
                            and max(ct_hist) - min(ct_hist) < 0.5
                            and ct >= 70):
                        # 读数恒定且异常偏高 -> 疑似 WMI 虚拟热区
                        ct_note = " [疑似虚拟热区读数]"
                print(f"[S5 {tag} step {step_global}] "
                      f"data_load={w_data / w_n * 1000:.1f}ms "
                      f"forward={w_fwd / w_n * 1000:.1f}ms "
                      f"opt={w_opt / w_n * 1000:.1f}ms "
                      f"loss={e_sum / e_n:.4f} util={u}% "
                      f"gpu_temp={gt}C cpu_temp={ct if ct is None else round(ct, 1)}C"
                      f"{ct_note} "
                      f"lr={lr_at(step_global):.2e}", flush=True)
                if gt is not None and gt >= GPU_TEMP_MAX:
                    print(f"[S5] GPU temp {gt}C >= {GPU_TEMP_MAX}C，暂停 20s 降温",
                          flush=True)
                    time.sleep(20)
                if (ct is not None and ct >= CPU_TEMP_MAX
                        and not ct_note):
                    cl = cpu_load()
                    if cl is not None and cl >= 70:
                        print(f"[S5] CPU temp {ct:.1f}C >= {CPU_TEMP_MAX}C "
                              f"且负载 {cl:.0f}%，暂停 20s 降温", flush=True)
                        time.sleep(20)
                    else:
                        print(f"[S5] CPU temp 读数 {ct:.1f}C 但负载仅 "
                              f"{cl if cl is not None else 'NA'}%，判定非真实过热，"
                              f"仅记录不暂停", flush=True)
                w_data = w_fwd = w_opt = w_n = 0.0
            if time.time() - t_start > MAX_TOTAL_SEC:
                print(f"[S5] 超过 {MAX_TOTAL_SEC / 60:.0f}min 绝对预算，提前收尾进入验证",
                      flush=True)
                break
            del logits, loss
        else:
            e_sec = time.time() - e_t0
            epoch_secs.append(round(e_sec, 1))
            print(f"[S5 {tag} EPOCH {epoch}/{epochs}] train_loss={e_sum / e_n:.4f} "
                  f"({e_sec:.0f}s, cum {time.time() - t_start:.0f}s)", flush=True)
            if epoch == 1 and epochs == EPOCHS and e_sec > EPOCH_BUDGET_SEC:
                print(f"[S5] epoch1 耗时 {e_sec:.0f}s > 预算 {EPOCH_BUDGET_SEC}s，"
                      f"按预案降 epochs {EPOCHS}->3", flush=True)
                epochs = 3
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            continue
        epoch_secs.append(round(time.time() - e_t0, 1))
        break

    val_ce, val_ppl = eval_full(model, val_ids, vocab_size)
    avg_util = util_sum / util_n if util_n else 0.0
    total_sec = round(time.time() - t_start, 1)
    comp = t_data_all + t_fwd_all + t_opt_all
    data_frac = t_data_all / comp if comp > 0 else 0.0
    print(f"[S5 {tag}] FINAL val_ppl={val_ppl:.4f} avg_gpu_util={avg_util:.1f}% "
          f"total={total_sec}s data_frac={data_frac * 100:.1f}%")
    return val_ce, val_ppl, avg_util, total_sec, epoch_secs, data_frac


def main():
    set_seed(SEED)
    vocab_size, train_ids, val_ids, chars = load_tinystories()
    print(f"[S5] tinystories 25MB vocab={vocab_size} train={train_ids.numel()} "
          f"val={val_ids.numel()}")

    fixed = FixedFFN_LM(d=128, vocab_size=vocab_size, L=4, width=WIDTH_E).to(DEVICE)
    n_params = count_params(fixed)
    print(f"[S5] Fixed params={n_params} (目标对齐 1.82M, "
          f"E={WIDTH_E})", flush=True)

    ce, cp, cu, ct, es, df = train_one(fixed, train_ids, val_ids, vocab_size,
                                       "Fixed")
    # ---- checkpoint：最终模型保存（不改训练逻辑）----
    ckpt_path = os.path.join(BASE, "checkpoints", f"stage5_fixed_{SEED}.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"state_dict": fixed.state_dict(),
                "cfg": dict(d=128, vocab_size=vocab_size, L=4, width=WIDTH_E,
                            seq_len=SEQ_LEN, batch_size=BATCH_SIZE,
                            epochs=EPOCHS, seed=SEED),
                "final_ce": ce, "final_ppl": cp}, ckpt_path)
    print(f"[S5] CKPT saved -> {ckpt_path}", flush=True)
    del fixed
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "val_ce", "val_ppl", "gpu_util_pct",
                    "sec", "epoch_secs", "data_frac", "note"])
        w.writerow(["FixedFFN_E14", n_params, round(ce, 4), round(cp, 4),
                    round(cu, 1), ct, ";".join(map(str, es)),
                    round(df, 4), "CellMoE 对照见 results_tinystories.csv"])
    print(f"[S5] CSV -> {OUT_CSV}")
    print(f"[S5] Fixed PPL={cp:.4f}")


if __name__ == "__main__":
    main()
