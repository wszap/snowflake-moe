# -*- coding: utf-8 -*-
"""任务：控制变量重跑 CellMoE（剥离温度暂停污染，得到真实计算效率对比）
- 用 Fixed MoE 训练时用的新版温度判定逻辑（带虚拟热区判定）重跑 CellMoE
- 配置完全不变：d=128, n_cells=4, n_organelles=8, n_memory=32, topk=4, L=4,
  batch=64, seq=128, 5 epoch, seed=2026, 25MB TinyStories
- 记录：每个 epoch 实际耗时、GPU 利用率、GPU/CPU 温度暂停次数、虚拟热区判定次数
- Fixed 对照行从 run_stage5_fixed.log 解析（同样的统计口径）
- 输出 output/results_tinystories_cellmoE_v2.csv
- 红线：不改架构、GPU < 80C、CPU < 85C（新版逻辑：CPU 需温度>=85 且负载>=70 才暂停）
"""
import csv
import gc
import math
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from run_cellmoe_ckpt import (  # noqa: E402
    DEVICE, SEQ_LEN, BATCH_SIZE, EPOCHS, LR, SEED, DIAG_EVERY,
    GPU_TEMP_MAX, LAMBDA_ENT, LAMBDA_MEM, LAMBDA_ORG,
    MAX_TOTAL_SEC, EPOCH_BUDGET_SEC, FastHierLM, count_params, eval_full,
    load_tinystories, set_seed, lm_batch, gpu_util, gpu_temp,
)

CPU_TEMP_MAX = 85   # run_cellmoe_ckpt 未定义，这里与 Fixed 脚本一致

OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output",
                                       "results_tinystories_cellmoE_v2.csv"))
FIXED_LOG = os.path.join(BASE, "run_stage5_fixed.log")


def cpu_temp():
    if DEVICE != "cuda":
        return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance -ClassName MSAcpi_ThermalZoneTemperature -Namespace root/wmi).CurrentTemperature"],
            capture_output=True, text=True, timeout=10)
        raw = out.stdout.strip()
        if not raw:
            return None
        # 单位 0.1 K -> Celsius
        return float(raw.splitlines()[0]) / 10.0 - 273.15
    except Exception:
        return None


def cpu_load():
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).LoadPercentage"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def parse_fixed_log(path):
    """从 run_stage5_fixed.log 解析 Fixed 的 epoch 耗时与温度统计（同口径对照）。"""
    stats = {"gpu_pauses": 0, "cpu_pauses": 0, "virtual_hot": 0,
             "non_real": 0, "epoch_secs": []}
    if not os.path.exists(path):
        return stats
    try:
        lines = open(path, "r", encoding="utf-8", errors="replace").readlines()
    except Exception:
        return stats
    for ln in lines:
        if "GPU temp" in ln and "暂停 20s" in ln:
            stats["gpu_pauses"] += 1
        elif "CPU temp" in ln and "暂停 20s" in ln:
            stats["cpu_pauses"] += 1
        if "疑似虚拟热区读数" in ln:
            stats["virtual_hot"] += 1
        if "判定非真实过热" in ln:
            stats["non_real"] += 1
        m = re.search(r"EPOCH \d+/\d+\].*?\((\d+)s", ln)
        if m:
            stats["epoch_secs"].append(int(m.group(1)))
    return stats


def train_cellmoe_v2():
    """CellMoE 重跑：Fixed 同款温度逻辑 + 暂停/虚拟热区计数。"""
    set_seed(SEED)
    vocab_size, train_ids, val_ids, chars = load_tinystories()
    model = FastHierLM(d=128, vocab_size=vocab_size, n_cells=4, n_organelles=8,
                       n_memory=32, topk_organelle=4, topk_cell=2, L=4).to(DEVICE)
    n_params = count_params(model)
    print(f"[V2] CellMoE params={n_params} train={train_ids.numel()} "
          f"val={val_ids.numel()}")

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
    gpu_pauses = 0
    cpu_pauses = 0
    virtual_hot = 0
    non_real = 0
    paused_sec = 0.0
    epoch_secs = []
    ct_hist = []
    w_data = w_fwd = w_opt = w_n = 0.0
    t_data_all = t_fwd_all = t_opt_all = 0.0
    prefetch = None

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
            logits, info = model(xb)
            t_fwd = time.time() - t0

            t0 = time.time()
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
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
                        ct_note = " [疑似虚拟热区读数]"
                        virtual_hot += 1
                print(f"[V2 CellMoE step {step_global}] "
                      f"data_load={w_data / w_n * 1000:.1f}ms "
                      f"forward={w_fwd / w_n * 1000:.1f}ms "
                      f"opt={w_opt / w_n * 1000:.1f}ms "
                      f"loss={e_sum / e_n:.4f} util={u}% "
                      f"gpu_temp={gt}C cpu_temp={ct if ct is None else round(ct, 1)}C"
                      f"{ct_note} "
                      f"lr={lr_at(step_global):.2e}", flush=True)
                if gt is not None and gt >= GPU_TEMP_MAX:
                    print(f"[V2] GPU temp {gt}C >= {GPU_TEMP_MAX}C，"
                          f"暂停 20s 降温", flush=True)
                    time.sleep(20)
                    gpu_pauses += 1
                    paused_sec += 20
                if (ct is not None and ct >= CPU_TEMP_MAX
                        and not ct_note):
                    cl = cpu_load()
                    if cl is not None and cl >= 70:
                        print(f"[V2] CPU temp {ct:.1f}C >= {CPU_TEMP_MAX}C "
                              f"且负载 {cl:.0f}%，暂停 20s 降温", flush=True)
                        time.sleep(20)
                        cpu_pauses += 1
                        paused_sec += 20
                    else:
                        print(f"[V2] CPU temp 读数 {ct:.1f}C 但负载仅 "
                              f"{cl if cl is not None else 'NA'}%，判定非真实过热，"
                              f"仅记录不暂停", flush=True)
                        non_real += 1
                w_data = w_fwd = w_opt = w_n = 0.0
            if time.time() - t_start > MAX_TOTAL_SEC:
                print(f"[V2] 超过 {MAX_TOTAL_SEC / 60:.0f}min 绝对预算，"
                      f"提前收尾进入验证", flush=True)
                break
            del logits, loss
        else:
            e_sec = time.time() - e_t0
            epoch_secs.append(round(e_sec, 1))
            print(f"[V2 CellMoE EPOCH {epoch}/{epochs}] "
                  f"train_loss={e_sum / e_n:.4f} "
                  f"({e_sec:.0f}s, cum {time.time() - t_start:.0f}s)", flush=True)
            if epoch == 1 and epochs == EPOCHS and e_sec > EPOCH_BUDGET_SEC:
                print(f"[V2] epoch1 耗时 {e_sec:.0f}s > 预算 {EPOCH_BUDGET_SEC}s，"
                      f"按预案降 epochs {EPOCHS}->3", flush=True)
                epochs = 3
                total_steps = n_steps * epochs
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            continue
        epoch_secs.append(round(time.time() - e_t0, 1))
        break

    val_ce, val_ppl = eval_full(model, val_ids, vocab_size)
    avg_util = util_sum / util_n if util_n else 0.0
    total_sec = round(time.time() - t_start, 1)
    pure_sec = round(total_sec - paused_sec, 1)
    print(f"[V2 CellMoE] FINAL val_ppl={val_ppl:.4f} avg_gpu_util={avg_util:.1f}% "
          f"total={total_sec}s pure={pure_sec}s paused={paused_sec}s "
          f"gpu_pauses={gpu_pauses} cpu_pauses={cpu_pauses} "
          f"virtual_hot={virtual_hot} non_real={non_real}")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 解析 Fixed 对照 ----
    fx = parse_fixed_log(FIXED_LOG)
    print(f"[V2 Fixed(log)] gpu_pauses={fx['gpu_pauses']} "
          f"cpu_pauses={fx['cpu_pauses']} virtual_hot={fx['virtual_hot']} "
          f"non_real={fx['non_real']} epoch_secs={fx['epoch_secs']}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "val_ppl", "avg_gpu_util_pct", "epoch_secs",
                    "total_sec", "paused_sec", "pure_compute_sec",
                    "gpu_pauses", "cpu_pauses", "virtual_hot", "non_real_hot",
                    "note"])
        w.writerow(["CellMoE_v2", round(val_ppl, 4), round(avg_util, 1),
                    ";".join(map(str, epoch_secs)), total_sec,
                    round(paused_sec, 1), pure_sec, gpu_pauses, cpu_pauses,
                    virtual_hot, non_real,
                    "本重跑：Fixed 同款温度判定逻辑（控制变量）"])
        w.writerow(["Fixed(对照log)", "", "",
                    ";".join(map(str, fx["epoch_secs"])), "", "", "",
                    fx["gpu_pauses"], fx["cpu_pauses"], fx["virtual_hot"],
                    fx["non_real"],
                    "历史 run_stage5_fixed.log 同口径统计"])
    print(f"[V2] CSV -> {OUT_CSV}")

    # ---- 验收 ----
    if len(epoch_secs) >= 3 and all(250 <= s <= 300 for s in epoch_secs[-3:]):
        print("[V2 PASS] CellMoE epoch 稳定在 250-300s -> 真实速度差约 1.2-1.4x，"
              "不是 1.8x")
    elif len(epoch_secs) >= 2:
        print("[V2 CHECK] CellMoE epoch 波动较大 -> 可能存在更深原因"
              "（如忆点检索随机性），需进一步分析")
    else:
        print("[V2 CHECK] 数据不足，无法稳定判断")


if __name__ == "__main__":
    train_cellmoe_v2()
