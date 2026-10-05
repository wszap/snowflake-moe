# -*- coding: utf-8 -*-
"""任务2：终身学习测试（CellMoE TinyStories 原 checkpoint）
- 加载 temp/checkpoints/cellmoe_tinystories.pt（原 1.0 倍 memory_read，PPL=10.0148）
- 每个细胞 add_new_memory(8)，冻结全部旧参数，只解冻新忆点（keys/asm/value）
- 新领域：圣经 KJV（bible_kjv.txt），用 TinyStories 字符表 tokenize，
  缺失字符（\r # % [ ] • TM）替换为空格
- 训练 3 epoch（batch=64, seq=128, lr=1e-3, seed=2026）
- 评估：旧领域（TS val）PPL + 新领域（圣经 val/test）PPL
- 输出 output/results_lifelong_tinystories.csv
- 红线：不改架构其他部分、GPU 温度 < 80C
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

from run_cellmoe_ckpt import (  # noqa: E402
    SEQ_LEN, load_tinystories, set_seed,
)

SEQ_LEN_ = SEQ_LEN
BATCH_SIZE, EPOCHS, LR, SEED = 64, 3, 1e-3, 2026
NEW_ADD = 8            # 每个细胞新增忆点数
GPU_TEMP_MAX = 80
CKPT_PATH = os.path.join(BASE, "checkpoints", "cellmoe_tinystories.pt")
LIFELONG_CKPT = os.path.join(BASE, "checkpoints", f"cellmoe_lifelong_{SEED}.pt")
BIBLE_PATH = os.path.join(BASE, "data", "bible_kjv.txt")
OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output",
                                       "results_lifelong_tinystories.csv"))


# ---------------- 带新忆点扩展槽的 FastCellMoE（等价 improved 2.3 逻辑） ----------------
class FastCellMoE_L(nn.Module):
    """FastCellMoE + add_new_memory 扩展（keys/asm/value 追加，None 时行为不变）。"""

    def __init__(self, d, n_organelles=8, n_memory=32, topk=4,
                 memory_read_scale=1.0, h=32):
        super().__init__()
        self.d = d
        self.n_organelles = n_organelles
        self.n_memory = n_memory
        self.topk = topk
        self.memory_read_scale = memory_read_scale
        self.encoder = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.W1 = nn.Parameter(torch.randn(n_organelles, d, h) * (2.0 / math.sqrt(d)))
        self.W2 = nn.Parameter(torch.randn(n_organelles, h, d) * (2.0 / math.sqrt(h)))
        self.memory_keys = nn.Parameter(torch.randn(n_memory, d) * 0.1)
        self.memory_assembly = nn.Parameter(torch.zeros(n_memory, n_organelles))
        self.memory_value = nn.Parameter(torch.zeros(n_memory, d))
        self.group_w = nn.Parameter(torch.tensor([0.5]))
        self.head = nn.Linear(d, d, bias=False)
        self.last_topk = None
        self.memory_attn = None
        self.new_memory_keys = None
        self.new_memory_assembly = None
        self.new_memory_value = None

    def add_new_memory(self, add=8, seed=2026):
        dev = self.memory_keys.device
        g = torch.Generator(device=dev).manual_seed(seed)
        new_keys = torch.randn(add, self.d, generator=g, device=dev) * 0.1
        new_asm = torch.zeros(add, self.n_organelles, device=dev)
        new_val = torch.zeros(add, self.d, device=dev)
        if self.new_memory_keys is None:
            self.new_memory_keys = nn.Parameter(new_keys)
            self.new_memory_assembly = nn.Parameter(new_asm)
            self.new_memory_value = nn.Parameter(new_val)
        else:
            self.new_memory_keys = nn.Parameter(
                torch.cat([self.new_memory_keys.detach(), new_keys], dim=0))
            self.new_memory_assembly = nn.Parameter(
                torch.cat([self.new_memory_assembly.detach(), new_asm], dim=0))
            self.new_memory_value = nn.Parameter(
                torch.cat([self.new_memory_value.detach(), new_val], dim=0))
        return [self.new_memory_keys, self.new_memory_assembly,
                self.new_memory_value], self.n_memory

    def forward(self, x):
        v = self.norm(self.encoder(x))
        if self.new_memory_keys is None:
            keys, asm, val = self.memory_keys, self.memory_assembly, self.memory_value
        else:
            keys = torch.cat([self.memory_keys, self.new_memory_keys], dim=0)
            asm = torch.cat([self.memory_assembly, self.new_memory_assembly], dim=0)
            val = torch.cat([self.memory_value, self.new_memory_value], dim=0)
        sim = v @ keys.T / (self.d ** 0.5)
        if self.new_memory_keys is not None:
            n_old = self.memory_keys.shape[0]
            sim_old, sim_new = sim[:, :n_old], sim[:, n_old:]
            if self.training:
                a_old = F.gumbel_softmax(sim_old, tau=1.0, hard=False, dim=-1)
                a_new = F.gumbel_softmax(sim_new, tau=1.0, hard=False, dim=-1)
            else:
                a_old = F.softmax(sim_old / 1.0, dim=-1)
                a_new = F.softmax(sim_new / 1.0, dim=-1)
            w_old = torch.sigmoid(self.group_w)
            attn = torch.cat([w_old * a_old, (1 - w_old) * a_new], dim=-1)
            attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-9)
        else:
            if self.training:
                attn = F.gumbel_softmax(sim, tau=1.0, hard=False, dim=-1)
            else:
                attn = F.softmax(sim / 1.0, dim=-1)
        self.memory_attn = attn.detach()
        assembly = attn @ asm
        weights = F.softmax(assembly, dim=-1)
        topk_w, topk_idx = torch.topk(weights, self.topk, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True).clamp_min(1e-9)
        org_out = torch.einsum("bd,ndh->bnh", x, self.W1)
        org_out = F.silu(org_out)
        org_out = torch.einsum("bnh,nhd->bnd", org_out, self.W2)
        w3 = topk_w.unsqueeze(-1)
        idx3 = topk_idx.unsqueeze(-1).expand(-1, -1, self.d)
        out = (org_out.gather(1, idx3) * w3).sum(1)
        out = out + self.memory_read_scale * (attn @ val)
        self.last_topk = topk_idx.detach()
        out = self.head(out)
        return out, {"weights": weights, "memory_attn": attn, "topk_idx": topk_idx}


class FastHierCellMoE_L(nn.Module):
    def __init__(self, d, n_cells=4, n_organelles=8, n_memory=32,
                 topk_organelle=4, topk_cell=2):
        super().__init__()
        self.d = d
        self.n_cells = n_cells
        self.topk_cell = topk_cell
        self.cells = nn.ModuleList([
            FastCellMoE_L(d, n_organelles=n_organelles, n_memory=n_memory,
                          topk=topk_organelle)
            for _ in range(n_cells)
        ])
        self.encoder = nn.Linear(d, d, bias=False)
        self.top_router = nn.Linear(d, n_cells, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        cell_outs, cell_infos = [], []
        for cell in self.cells:
            out, info = cell(x)
            cell_outs.append(out)
            cell_infos.append(info)
        cell_outs = torch.stack(cell_outs, dim=1)
        v = self.norm(self.encoder(x))
        gate_logits = self.top_router(v)
        if self.training:
            gate = F.gumbel_softmax(gate_logits, tau=1.0, hard=False, dim=-1)
        else:
            gate = F.softmax(gate_logits, dim=-1)
        topk_w, topk_idx = torch.topk(gate, self.topk_cell, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True).clamp_min(1e-9)
        gate_sparse = torch.zeros_like(gate)
        gate_sparse.scatter_(1, topk_idx, topk_w)
        out = (gate_sparse.unsqueeze(-1) * cell_outs).sum(dim=1)
        entropy = float(-(gate * (gate + 1e-9).log()).sum(-1).mean().item())
        return out, {"gate": gate, "gate_entropy": entropy,
                     "cell_infos": cell_infos}


class FastHierLM_L(nn.Module):
    def __init__(self, d, vocab_size, n_cells=4, n_organelles=8, n_memory=32,
                 topk_organelle=4, topk_cell=2, L=4):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d)
        self.in_proj = nn.Linear(d, d, bias=False)
        self.layers = nn.ModuleList([
            FastHierCellMoE_L(d, n_cells=n_cells, n_organelles=n_organelles,
                              n_memory=n_memory, topk_organelle=topk_organelle,
                              topk_cell=topk_cell)
            for _ in range(L)
        ])
        self.head = nn.Linear(d, vocab_size)

    def forward(self, tokens):
        x = self.in_proj(self.embed(tokens))
        info = None
        for layer in self.layers:
            B, T, d = x.shape
            y, li = layer(x.reshape(B * T, d))
            info = li if info is None else info
            x = y.reshape(B, T, d)
        return self.head(x), info


# ---------------- 工具函数 ----------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lm_batch(ids, batch_size, seq_len, seed):
    g = torch.Generator().manual_seed(seed)
    n = ids.numel() - seq_len - 1
    idx = torch.randint(0, n, (batch_size,), generator=g)
    offsets = idx.unsqueeze(1) + torch.arange(seq_len)
    xb = ids[offsets]
    yb = ids[offsets + 1]
    return xb.to(DEVICE), yb.to(DEVICE)


def eval_full(model, ids, vocab_size, batch_size=128):
    model.eval()
    k = batch_size * SEQ_LEN_
    n = ids.numel()
    total_ce, total_n = 0.0, 0
    with torch.no_grad():
        s = 0
        while s + k + 1 <= n:
            x = ids[s:s + k].view(batch_size, SEQ_LEN_).to(DEVICE)
            y = ids[s + 1:s + k + 1].view(batch_size, SEQ_LEN_).to(DEVICE)
            logits, _ = model(x)
            ce = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
            total_ce += ce * k
            total_n += k
            s += k
    model.train()
    avg_ce = total_ce / total_n
    return avg_ce, math.exp(avg_ce)


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


def tokenize_bible(path, stoi, val_frac=0.1, seed=SEED):
    """用 TinyStories 字符表 tokenize 圣经 KJV；缺失字符替换为空格。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    missing = sorted(set(text) - set(stoi.keys()))
    for ch in missing:
        text = text.replace(ch, " ")
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_val = int(ids.numel() * val_frac)
    rng = random.Random(seed)
    val_start = rng.randint(0, ids.numel() - n_val - 1)
    val_ids = ids[val_start:val_start + n_val]
    train_ids = torch.cat([ids[:val_start], ids[val_start + n_val:]])
    return train_ids, val_ids, missing


def new_memory_utilization(model, ids, vocab_size, n_memory=32,
                           n_batch=20, batch_size=16):
    """路由测试：新领域输入上，忆点检索 argmax 落在新忆点区间(>=n_memory)的比例。"""
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for _ in range(n_batch):
            xb, _ = lm_batch(ids, batch_size, SEQ_LEN_, seed=SEED + _)
            _, info = model(xb)
            for ci in info['cell_infos']:
                am = ci['memory_attn']
                hits += int((am.argmax(-1) >= n_memory).sum().item())
                total += am.shape[0]
    model.train()
    return hits / max(1, total)


def main():
    set_seed(SEED)
    vocab_size, train_ids, val_ids, chars = load_tinystories()
    stoi = {c: i for i, c in enumerate(chars)}

    # ---- 圣经数据（TS 词表 tokenize） ----
    if not os.path.exists(BIBLE_PATH):
        raise FileNotFoundError(f"缺少圣经数据 {BIBLE_PATH}")
    bible_train, bible_val, missing = tokenize_bible(BIBLE_PATH, stoi)
    print(f"[LIFE] vocab={vocab_size} ts_train={train_ids.numel()} "
          f"ts_val={val_ids.numel()} bible_train={bible_train.numel()} "
          f"bible_val={bible_val.numel()} missing_chars={missing}")

    # ---- 加载原 ckpt（FastHierLM 架构） ----
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    cfg = ckpt["cfg"]
    model = FastHierLM_L(d=cfg["d"], vocab_size=cfg["vocab_size"],
                         n_cells=cfg["n_cells"], n_organelles=cfg["n_organelles"],
                         n_memory=cfg["n_memory"],
                         topk_organelle=cfg["topk_organelle"],
                         topk_cell=cfg["topk_cell"], L=cfg["L"]).to(DEVICE)
    # 容忍新增的 group_w（旧 ckpt 无此参数，用初始化值 0.5）
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        print(f"[LIFE] missing keys (expected for new params): {missing}")
    print(f"[LIFE] ckpt loaded, recorded final_ppl={ckpt['final_ppl']:.4f}")

    # ---- 基线评估 ----
    _, base_ppl = eval_full(model, val_ids, vocab_size)
    _, base_bible_ppl = eval_full(model, bible_val, vocab_size)
    print(f"[LIFE BASELINE] ts_ppl={base_ppl:.4f} bible_ppl={base_bible_ppl:.4f}")

    # ---- 加新忆点 + 冻结旧参数 ----
    n_cells = 0
    for layer in model.layers:
        for cell in layer.cells:
            cell.add_new_memory(NEW_ADD, seed=SEED)
            n_cells += 1
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in model.layers:
        for cell in layer.cells:
            cell.new_memory_keys.requires_grad_(True)
            cell.new_memory_assembly.requires_grad_(True)
            cell.new_memory_value.requires_grad_(True)
            cell.group_w.requires_grad_(True)
    trainable = count_params(model)
    print(f"[LIFE] 冻结完成：{n_cells} 个细胞各 +{NEW_ADD}，"
          f"可训练参数={trainable}（新忆点 keys/asm/val）")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---- 只训练新忆点：圣经 3 epoch ----
    key_params = [p for name, p in model.named_parameters()
                  if p.requires_grad and ("new_memory_keys" in name
                                          or "new_memory_assembly" in name
                                          or "group_w" in name)]
    val_params = [p for name, p in model.named_parameters()
                  if p.requires_grad and "new_memory_value" in name]
    opt = torch.optim.Adam([
        {"params": key_params, "lr": LR, "weight_decay": 1e-2},
        {"params": val_params, "lr": 3e-6, "weight_decay": 1e-1},
    ])
    n_steps = max(1, bible_train.numel() // (SEQ_LEN_ * BATCH_SIZE))
    t0 = time.time()
    step_global = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        e_sum, e_n = 0.0, 0
        for _ in range(n_steps):
            xb, yb = lm_batch(bible_train, BATCH_SIZE, SEQ_LEN_,
                              seed=SEED + step_global)
            opt.zero_grad(set_to_none=True)
            logits, info = model(xb)
            ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            loss = ce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step_global += 1
            e_sum += loss.item()
            e_n += 1
            del logits, loss
            if step_global % 200 == 0:
                temp = gpu_temp()
                if temp is not None and temp >= GPU_TEMP_MAX:
                    print(f"[LIFE] GPU temp {temp}C >= {GPU_TEMP_MAX}C，"
                          f"暂停 20s 降温", flush=True)
                    time.sleep(20)
        print(f"[LIFE EPOCH {epoch}/{EPOCHS}] train_loss={e_sum / e_n:.4f} "
              f"({(time.time() - t0):.0f}s)", flush=True)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    os.makedirs(os.path.dirname(LIFELONG_CKPT), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg,
                "base_ts_ppl": base_ppl, "base_bible_ppl": base_bible_ppl,
                "new_add": NEW_ADD}, LIFELONG_CKPT)
    print(f"[LIFE] SAVED -> {LIFELONG_CKPT}")

    # ---- 训练后评估 ----
    _, life_ppl = eval_full(model, val_ids, vocab_size)
    _, life_bible_ppl = eval_full(model, bible_val, vocab_size)
    util = new_memory_utilization(model, bible_val, vocab_size,
                                  n_memory=cfg["n_memory"])
    regress = life_ppl - base_ppl
    drop = (base_bible_ppl - life_bible_ppl) / max(1e-9, base_bible_ppl)
    print(f"[LIFE FINAL] ts_ppl={life_ppl:.4f} (base {base_ppl:.4f}, "
          f"退化{regress:+.4f})  bible_ppl={life_bible_ppl:.4f} "
          f"(base {base_bible_ppl:.4f}, 下降{drop:.2%})  "
          f"new_mem_util={util:.3%}")

    # ---- CSV ----
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "ts_base_ppl", "ts_life_ppl", "ts_regress",
                    "bible_base_ppl", "bible_life_ppl", "bible_drop",
                    "new_mem_util", "pass"])
        w.writerow(["lifelong", round(base_ppl, 4), round(life_ppl, 4),
                    round(regress, 4), round(base_bible_ppl, 4),
                    round(life_bible_ppl, 4), round(drop, 4),
                    round(util, 4), 1 if (regress < 0.5 and drop > 0.2) else 0])
    print(f"[LIFE] CSV -> {OUT_CSV}")

    # ---- 验收 ----
    ok_regress = regress < 0.5
    ok_drop = drop > 0.20
    if ok_regress and ok_drop:
        print(f"[LIFE PASS] 旧领域退化{regress:.4f}<0.5 且 "
              f"新领域下降{drop:.2%}>20%")
    else:
        print(f"[LIFE FAIL] 退化{regress:.4f}"
              f"{'' if ok_regress else ' 不达标'} / "
              f"下降{drop:.2%}{'' if ok_drop else ' 不达标'}")


if __name__ == "__main__":
    main()
