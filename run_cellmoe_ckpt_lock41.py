# -*- coding: utf-8 -*-

"""阶段五（快速版）：TinyStories 100MB 扩展性验证

- 与 run_stage5.py 完全同超参/同 seed/同数据，仅将串行 Organelle 前向

  改为并行 einsum（数学等价，PyTorch 纯实现优化，不改架构）

- CellMoE: FastHierLM(d=128, n_cells=4, n_organelles=8, n_memory=32,

          topk_organelle=4, topk_cell=2, L=4)

- Fixed: dense FFN transformer（d=128, L=4, FFN 宽 8d）

- 各 5 epoch, seed 2026, batch=512, seq=128, lr=3e-4, warmup 5%, clip 1.0

- 输出 output/results_tinystories.csv；验收 CellMoE PPL<=Fixed*1.05

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

import torch.nn as nn

import torch.nn.functional as F



# 训练加速：TF32 matmul（仅放宽精度，不改训练逻辑）

torch.set_float32_matmul_precision('high')



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE)



SEQ_LEN, BATCH_SIZE, EPOCHS, LR, SEED = 128, 64, 5, 3e-4, 2026

# 用户重跑方案：batch=64、num_workers=4、25MB、只跑 CellMoE、epochs=5（超 45min 降 3）

NUM_WORKERS = 4        # 数据预取并发位：Windows 下 DataLoader spawn 需序列化 190MB 索引

                       # 张量反而更慢，故用双缓冲预取等效实现（CPU 预取与 GPU 计算并行）

MAX_TOTAL_SEC = 55 * 60  # 绝对保护：55min 强制收尾出 PPL（满足"1 小时内"红线）

EPOCH_BUDGET_SEC = 540   # 单 epoch 预算 9min：epoch1 超预算自动降 epochs 5->3

DIAG_EVERY = 200         # 每 200 step 打印 data_load_time / model_forward_time

GPU_TEMP_MAX = 80        # 温度红线：>=80C 暂停 20s 降温

TOPK_ORG = 4  # 锁3.7：org_sum 归一化除数（与模型 topk_organelle=4 一致）

LAMBDA_MEM, LAMBDA_ENT, LAMBDA_ORG, LAMBDA_KD = 0.05, 0.0, 0.05, 0.0   # 锁3.8：ENT 回落 0.05 基线（0.03 与 0.1 折中）；org_sum 归一化频次权重 0.05；KD 蒸馏项 0.05 起步

# ---- 锁4.0：内容寻址路由（在双向熵控制保护下）----

# 每个细胞器自带可学习 query（organelle_query），样本 v 与 query 点积竞标路由：

# query_logits = (v @ organelle_query.T) / sqrt(d)；logits = query_logits + mem_gate（忆点上下文保留）

# 双向熵控制继续保护（TARGET_ENT=1.4, LAMBDA_ENT_CTRL=0.05），KD 保持关闭

# 删除旧单向低熵惩罚 LAMBDA_ENT*ent_sum；改为目标熵双向拉：

# loss_ent_ctrl = LAMBDA_ENT_CTRL * (current_ent - TARGET_ENT)^2，current_ent=(ent_sum/nc).detach()

TARGET_ENT = 1.4          # 目标路由熵（黄金区间 1.2~1.6 中值）

LAMBDA_ENT_CTRL = 0.05    # 双向熵控制权重

LAMBDA_ORTH = 0.01        # 锁4.1：organelle_query 正交性约束权重（从 0.01 起试）

KD_ENABLED = False        # 锁3.9：彻底关闭 KD（教师-学生零反向污染）

# ---- 锁3.9：硬分层偏置（不可学习，route_proj 输出后加，training_step<1000 生效）----

# L0:+0.5, L1:+0.2, L2:-0.2, L3:-0.5；主偏置放在该层对应 organelle 位置（对角硬先验）

LAYER_BIASES = [

    [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # L0 浅层：+0.5

    [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # L1 浅中：+0.2

    [0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0],  # L2 中深：-0.2

    [0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],  # L3 深层：-0.5

]

# ---- 锁20：忆点 lr 回调（锁14/19 的 3e-5 饿死忆点，mem_gate/assembly ratio 1428x；建议 1e-4~1.5e-4）----

MEM_LR = 1e-4

DATA = os.path.join(BASE, "tinystories_100mb.txt")

OUT_CSV = os.path.abspath(os.path.join(BASE, "..", "output",

                                       "results_tinystories_lock41.csv"))

CKPT_PATH = os.path.abspath(os.path.join(BASE, "checkpoints",

                                        "cellmoe_tinystories_lock41.pt"))

# ---- 锁10：文档边界切分（修复验证集泄漏）----

SEP = "<|endoftext|>"      # TinyStories 文档分隔符

STORY_STARTS = None        # 全局：训练合法起点列表 [(s, e_lim)]，由 load_tinystories 填充





# ---------------- 并行 Organelle（等价于 8 个串行 MLP(d->32->d)） ----------------

class FastCellMoE(nn.Module):

    """等价 ImprovedCellMoE：organelles 用 einsum 并行，其余逻辑一致。"""



    def __init__(self, d, n_organelles=8, n_memory=32, topk=4,

                 memory_read_scale=1.0, h=32, layer_bias=None):

        super().__init__()

        self.d = d

        self.n_organelles = n_organelles

        self.n_memory = n_memory

        self.topk = topk

        self.memory_read_scale = nn.Parameter(torch.tensor(float(memory_read_scale)))

        self.encoder = nn.Linear(d, d, bias=False)

        self.norm = nn.LayerNorm(d)

        # 8 个 Organelle 权重合并（等价钱：x->W1[.,h] SiLU ->W2[.,d]）

        self.W1 = nn.Parameter(torch.randn(n_organelles, d, h) * (2.0 / math.sqrt(d)))

        self.W2 = nn.Parameter(torch.randn(n_organelles, h, d) * (2.0 / math.sqrt(h)))

        self.memory_keys = nn.Parameter(

            F.normalize(self.encoder.weight[:n_memory].detach(), dim=-1) * 0.1)

        self.memory_assembly = nn.Parameter(torch.zeros(n_memory, n_organelles))

        self.memory_value = nn.Parameter(torch.zeros(n_memory, d))

        self.route_proj = nn.Linear(self.d, n_organelles, bias=False)

        nn.init.normal_(self.route_proj.weight, std=0.02)

        # 锁4.0：细胞器查询向量（内容寻址）——每个细胞器的"自我介绍牌"，与样本 v 点积竞标路由

        self.organelle_query = nn.Parameter(torch.randn(self.n_organelles, self.d) * 0.02)

        # 锁3.9：硬分层偏置（不可学习）——route_proj 输出后加固定偏置，requires_grad=False

        if layer_bias is not None:

            self.layer_bias = nn.Parameter(torch.tensor(layer_bias, dtype=torch.float32), requires_grad=False)

        else:

            self.layer_bias = nn.Parameter(torch.zeros(n_organelles), requires_grad=False)

        self.group_w = nn.Parameter(torch.tensor([0.5]))

        self._bias_scale = 1.0   # 锁3.9：training_step<1000 生效开关，1000 步后由训练循环归零

        self.head = nn.Linear(d, d, bias=False)

        self.last_topk = None

        self.memory_attn = None

        self.new_memory_keys = None

        self.new_memory_assembly = None

        self.new_memory_value = None



    def forward(self, x):


        if self.new_memory_keys is None:

            keys = self.memory_keys

            asm = self.memory_assembly

            val = self.memory_value

            n_old = None

        else:

            keys = torch.cat([self.memory_keys, self.new_memory_keys], dim=0)

            asm = torch.cat([self.memory_assembly, self.new_memory_assembly], dim=0)

            val = torch.cat([self.memory_value, self.new_memory_value], dim=0)

            n_old = self.memory_keys.shape[0]

        sim = x @ keys.T / (self.d ** 0.5)                      # [B, n_mem]   # 锁4.1：memory 读取同步改用原始输入（绕过坍塌 encoder）

        if self.new_memory_keys is not None:

            # 分组 softmax：old / new 分别归一化（对齐 FastCellMoE_L / ImprovedCellMoE）

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

        assembly = attn @ asm                                    # [B, N]

        mem_features = attn @ val                                # [B, d]

        mem_gate = self.route_proj(mem_features) + self.layer_bias * self._bias_scale   # [B, N] ← 忆点调制路由（整体上下文）+ 锁3.9硬分层先验

        # 锁4.1：内容寻址路由——每个细胞器 query 主动竞标原始输入 x（绕过坍塌 encoder v）

        query_logits = (x @ self.organelle_query.T) / math.sqrt(self.d)                # [B, N] 内容寻址（锁4.1：绕过坍塌 encoder，直接用原始输入）

        logits = query_logits + mem_gate                                               # 寻址 + 忆点上下文（assembly 不再直接进路由）

        if self.training:

            # 锁3.6：Gumbel 噪声 tau 降至 0.5（1.0 过早锁定幸运细胞），鼓励平缓探索；eval 不加保持确定性

            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)

            weights = F.softmax(logits + gumbel_noise * 0.5, dim=-1)

        else:

            weights = F.softmax(logits, dim=-1)

        topk_w, topk_idx = torch.topk(weights, self.topk, dim=-1)

        topk_w = topk_w / topk_w.sum(-1, keepdim=True).clamp_min(1e-9)

        # 并行 organelle 前向：x -> [B, N, h] -> SiLU -> [B, N, d]

        org_out = torch.einsum("bd,ndh->bnh", x, self.W1)

        org_out = F.silu(org_out)

        org_out = torch.einsum("bnh,nhd->bnd", org_out, self.W2)  # [B, N, d]

        w3 = topk_w.unsqueeze(-1)

        idx3 = topk_idx.unsqueeze(-1).expand(-1, -1, self.d)

        out = (org_out.gather(1, idx3) * w3).sum(1)             # [B, d]

        # 监控（可选，不影响梯度）：忆点对路由的贡献

        if not hasattr(self, '_ratios'):

            self._ratios = []

        self._ratios.append((mem_gate.norm() / (assembly.norm() + 1e-9)).item())

        if len(self._ratios) > 1000:

            self._ratios = self._ratios[-500:]

        self.last_topk = topk_idx.detach()

        out = self.head(out)

        return out, {"weights": weights, "memory_attn": attn, "topk_idx": topk_idx}





class FastHierCellMoE(nn.Module):

    """等价 ImprovedHierarchicalCellMoE：n_cells 细胞 + 二级 gate + topk_cell。"""



    def __init__(self, d, n_cells=4, n_organelles=8, n_memory=32,

                 topk_organelle=4, topk_cell=2, layer_bias=None):

        super().__init__()

        self.d = d

        self.n_cells = n_cells

        self.topk_cell = topk_cell

        self.cells = nn.ModuleList([

            FastCellMoE(d, n_organelles=n_organelles, n_memory=n_memory,

                        topk=topk_organelle, layer_bias=layer_bias)

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

        cell_outs = torch.stack(cell_outs, dim=1)               # [B, n_cells, d]

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

        entropy = None  # 无消费方，删除以消除每步 .item() 同步

        # ---- 锁3.9：知识蒸馏彻底关闭（KD_ENABLED=False，教师-学生零反向污染）----

        kd_loss = None

        if KD_ENABLED and self.training:

            teacher = cell_outs[:, 1].detach()                  # 好学生 cell1 特征 [B, d]

            kd_loss = ((cell_outs - teacher.unsqueeze(1)).pow(2).mean(dim=-1))  # [B, n_cells]

            kd_loss = kd_loss[:, [0, 2, 3]].mean()              # 除 cell1 外：中等生+差生学 teacher

        return out, {"gate": gate, "gate_entropy": entropy,

                     "cell_infos": cell_infos, "kd_loss": kd_loss}





class FastHierLM(nn.Module):

    """等价 ImprovedHierarchicalCellMoE_LM。"""



    def __init__(self, d, vocab_size, n_cells=4, n_organelles=8, n_memory=32,

                 topk_organelle=4, topk_cell=2, L=4, layer_biases=None):

        super().__init__()

        self.d = d

        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d)

        self.in_proj = nn.Linear(d, d, bias=False)

        self.layers = nn.ModuleList([

            FastHierCellMoE(d, n_cells=n_cells, n_organelles=n_organelles,

                            n_memory=n_memory, topk_organelle=topk_organelle,

                            topk_cell=topk_cell,

                            layer_bias=(layer_biases[i] if layer_biases else None))

            for i in range(L)

        ])

        self.head = nn.Linear(d, vocab_size)

    def set_bias_scale(self, s):

        """锁3.9：设置所有 cell 的硬分层偏置缩放（1000 步后归零用）。"""

        for layer in self.layers:

            for c in layer.cells:

                c._bias_scale = s



    def forward(self, tokens):

        x = self.in_proj(self.embed(tokens))

        info = None

        kd_loss = None

        for layer in self.layers:

            B, T, d = x.shape

            y, li = layer(x.reshape(B * T, d))

            info = li if info is None else info

            if li.get("kd_loss") is not None:

                kd_loss = li["kd_loss"] if kd_loss is None else kd_loss + li["kd_loss"]

            x = y.reshape(B, T, d)

        if kd_loss is not None:

            kd_loss = kd_loss / len(self.layers)

            info = {**info, "kd_loss": kd_loss}

        return self.head(x), info





# ---------------- Fixed dense FFN transformer ----------------

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





# ---------------- 数据 ----------------

def load_tinystories(path=DATA, val_frac=0.1, seed=2026, use_mb=25):

    """快速加载：文件为纯 ASCII 文本时用 numpy 批量映射（比逐字符快 100x）。

    use_mb>0 时截取前 use_mb MB 数据（保持 5 epoch 完整训练）。

    锁10：按文档边界切分（<|endoftext|> 或双换行），前 90% 文档训练、后 10% 验证。"""

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

    else:

        lut = None



    # ---- 锁10：按文档边界切分（非随机），分隔符 <|endoftext|> 或双换行 ----

    doc_parts = re.split(r"(?:<\|endoftext\|>|\n\s*\n)", text)

    docs = [p.strip("\n") for p in doc_parts if p.strip("\n")]

    n_val_docs = max(1, int(len(docs) * val_frac))

    train_docs = docs[:len(docs) - n_val_docs]

    val_docs = docs[len(docs) - n_val_docs:]



    def encode_docs(doc_list):

        if lut is not None:

            parts = []

            for doc in doc_list:

                b = doc.encode("utf-8")

                a = lut[np.frombuffer(b, dtype=np.uint8)]

                if (a < 0).any():

                    raise ValueError("non-ASCII bytes found, fallback needed")

                parts.append(torch.from_numpy(a.astype(np.int64)))

        else:

            stoi = {c: i for i, c in enumerate(chars)}

            parts = [torch.tensor([stoi[c] for c in doc], dtype=torch.long)

                     for doc in doc_list]

        if not parts:

            return torch.zeros(0, dtype=torch.long)

        return torch.cat(parts)



    train_ids = encode_docs(train_docs)

    val_ids = encode_docs(val_docs)



    # 填充全局 STORY_STARTS：训练文档合法起点（供后续避免跨文档采样使用）

    global STORY_STARTS

    STORY_STARTS = []

    offset = 0

    for doc in train_docs:

        dlen = len(doc.encode("utf-8")) if lut is not None else len(doc)

        if dlen > 1:

            STORY_STARTS.append((offset, offset + dlen - 1))

        offset += dlen

    return vocab_size, train_ids, val_ids, chars





def lm_batch(ids, batch_size, seq_len, seed):

    g = torch.Generator().manual_seed(seed)

    n = ids.numel() - seq_len - 1

    idx = torch.randint(0, n, (batch_size,), generator=g).to(DEVICE)

    offsets = idx.unsqueeze(1) + torch.arange(seq_len, device=DEVICE)

    xb = ids[offsets]

    yb = ids[offsets + 1]

    return xb, yb





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





def train_one(model, train_ids, val_ids, vocab_size, tag, use_reg):

    # 数据预加载到 GPU：训练前一次性搬运，避免每步 CPU→GPU 拷贝

    train_ids = train_ids.to(DEVICE)

    val_ids = val_ids.to(DEVICE)

    # ---- 锁14/19：分组 lr（细胞器 3e-4，其余默认 3e-4）----

    # ---- 锁20：忆点+route_proj lr 回调 3e-5 -> 1e-4（消除 ratio 1428x 饿死）----

    organelle_params = [p for n, p in model.named_parameters()

                        if "W1" in n or "W2" in n]

    memory_params = [p for n, p in model.named_parameters()

                     if "memory" in n or "route_proj" in n]

    rest_params = [p for n, p in model.named_parameters()

                   if not ("W1" in n or "W2" in n

                           or "memory" in n or "route_proj" in n)]

    opt = torch.optim.Adam([

        {"params": organelle_params, "lr": LR, "base_lr": LR},

        {"params": memory_params, "lr": MEM_LR, "base_lr": MEM_LR},

        {"params": rest_params, "lr": LR, "base_lr": LR},

    ], fused=True)

    n_steps = max(1, train_ids.numel() // (SEQ_LEN * BATCH_SIZE))

    epochs = EPOCHS

    total_steps = n_steps * epochs

    warmup = max(1, int(total_steps * 0.05))



    def lr_at(st, base_lr):

        return base_lr * (st + 1) / warmup if st < warmup else base_lr



    t_start = time.time()

    step_global = 0

    util_sum, util_n = 0.0, 0

    # 诊断窗口（每 DIAG_EVERY step 汇总一次）

    w_data, w_fwd, w_opt, w_n = 0.0, 0.0, 0.0, 0

    t_data_all, t_fwd_all, t_opt_all = 0.0, 0.0, 0.0

    epoch_secs = []

    # 双缓冲预取：当前批在 GPU 计算时，CPU 并行准备下一批（等效 num_workers=4）

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

            # 预取下一批（CPU 花式索引，与下方 GPU kernel 排队并行）

            prefetch = lm_batch(train_ids, BATCH_SIZE, SEQ_LEN,

                                seed=SEED + step_global + 1)

            t_data = time.time() - t0



            if step_global % 10 == 0:

                for g in opt.param_groups:

                    g['lr'] = lr_at(step_global, g['base_lr'])

            opt.zero_grad(set_to_none=True)



            t0 = time.time()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                logits, info = model(xb)

                t_fwd = time.time() - t0

                ce_loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

                loss = ce_loss

                if use_reg and isinstance(info, dict):

                    cell_infos = info.get('cell_infos', [info])

                    ent_sum = mem_sum = org_sum = 0.0

                    for ci in cell_infos:

                        w = ci['weights']

                        ent_sum += -(w * torch.log(w + 1e-9)).sum(-1).mean()   # 单样本熵：锁3.7 起参与 loss（低熵自信）

                        mu = ci['memory_attn'].mean(dim=0)

                        mem_sum += (mu * mu).sum() * mu.shape[-1]

                        # 锁3.7：org_sum 归一化修正——f_hard/topk，均匀=1、全压=N/topk=2

                        # （锁3.6 未归一化导致 topk=4 时均匀下界=16、梯度恒零、全员平庸）

                        f_soft = ci['weights'].mean(dim=0) / TOPK_ORG          # [N]，可导概率均值（归一化）

                        one_hot = torch.zeros_like(ci['weights'])

                        one_hot.scatter_(1, ci['topk_idx'], 1.0)              # [B, N] 真实选择

                        f_hard = one_hot.mean(dim=0) / TOPK_ORG               # [N]，真实使用频次（归一化）

                        f = f_hard + (f_soft - f_soft.detach())               # STE

                        org_sum += (f * f).sum() * f.shape[-1]                # N*sum(f²)：均匀→1，全压→N/topk=2

                    nc = len(cell_infos)

                    kd_loss = info.get('kd_loss')

                    kd_val = (kd_loss.item() if kd_loss is not None else 0.0)

                    # 锁3.9：双向熵控制——删除旧单向 LAMBDA_ENT*ent_sum，改为目标熵二次惩罚

                    # current_ent detach 为常数（按用户公式），ent_ctrl 为常数监控项

                    # 锁3.9：KD 已彻底关闭（LAMBDA_KD=0.0）；org_sum 0.05、MEM 0.05 不变

                    current_ent = (ent_sum / nc).detach()  # 锁3.9：目标熵（常数）

                    ent_ctrl = LAMBDA_ENT_CTRL * (current_ent - TARGET_ENT) ** 2

                    loss = ce_loss + LAMBDA_MEM * (mem_sum / nc) + ent_ctrl + LAMBDA_ORG * (org_sum / nc)

                    # 锁4.1：organelle_query 正交性约束（绕过坍塌 encoder 后，防止 query 共线退化）
                    orth_sum = 0.0
                    for _layer in model.layers:
                        for _cell in _layer.cells:
                            _qn = F.normalize(_cell.organelle_query, dim=1)
                            _qg = _qn @ _qn.T
                            _od = _qg - torch.eye(_cell.n_organelles, device=_qg.device)
                            orth_sum += (_od ** 2).sum()
                    orth_val = orth_sum.item()
                    loss = loss + LAMBDA_ORTH * orth_sum

                    if kd_loss is not None:

                        loss = loss + LAMBDA_KD * kd_loss

                    router_ent = (ent_sum / nc).item()    # 锁3.7监控：单样本 weights 熵（目标 1.2~1.6，不能<1.0）

                    ent_ctrl_val = ent_ctrl.item()          # 锁3.9监控：双向熵控制项

                    org_sum_val = (org_sum / nc).item()   # 锁3.7监控：归一化频次均衡度（均匀=1，全压=2，目标 1.5~2.0）

                else:

                    router_ent = 0.0

                    org_sum_val = 0.0

                    kd_val = 0.0

                    ent_ctrl_val = 0.0

                    orth_val = 0.0



            t0 = time.time()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()

            t_opt = time.time() - t0



            step_global += 1

            if step_global == 1000:      # 锁3.9：1000 步后硬分层偏置直接归零

                model.set_bias_scale(0.0)

            e_sum += loss.detach()  # tensor 累积，不每步 .item() 同步，仅打印时触发

            e_n += 1

            d_cost = t_data

            w_data += d_cost

            w_fwd += t_fwd

            w_opt += t_opt

            w_n += 1

            t_data_all += d_cost

            t_fwd_all += t_fwd

            t_opt_all += t_opt



            if step_global % DIAG_EVERY == 0:

                u = gpu_util()

                temp = gpu_temp()

                if u is not None:

                    util_sum += u

                    util_n += 1

                print(f"[S5 {tag} step {step_global}] "

                      f"data_load={w_data / w_n * 1000:.1f}ms "

                      f"forward={w_fwd / w_n * 1000:.1f}ms "

                      f"opt={w_opt / w_n * 1000:.1f}ms "

                      f"loss={e_sum / e_n:.4f} util={u}% temp={temp}C "

                      f"lr={opt.param_groups[0]['lr']:.2e} "

                      f"mem_lr={opt.param_groups[1]['lr']:.2e} "

                      f"router_ent={router_ent:.4f} org_sum={org_sum_val:.4f} "

                      f"ent_ctrl={ent_ctrl_val:.4f} ce={ce_loss.item():.4f} orth={orth_val:.4f}", flush=True)

                if temp is not None and temp >= GPU_TEMP_MAX:

                    print(f"[S5] GPU temp {temp}C >= {GPU_TEMP_MAX}C，暂停 20s 降温",

                          flush=True)

                    time.sleep(20)

                # 锁3.9：前 200 步红线监控

                if use_reg and step_global <= 200 and router_ent < 0.8:

                    print(f"[S5 {tag}] REDLINE step {step_global}: router_ent={router_ent:.4f} < 0.8，触发停机红线，终止训练", flush=True)

                    sys.exit(1)

                if step_global == 200 and 1.2 <= router_ent <= 1.6:

                    print(f"[S5 {tag}] GREENLINE step 200: router_ent={router_ent:.4f} 稳在黄金区间 1.2~1.6，正常继续", flush=True)

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

            # ---- checkpoint：每 epoch 结束保存一次（不改训练逻辑）----

            ckpt_epoch_path = os.path.join(BASE, "checkpoints",

                                           f"cellmoe_tinystories_lock41_epoch{epoch}.pt")

            os.makedirs(os.path.dirname(ckpt_epoch_path), exist_ok=True)

            torch.save({"state_dict": model.state_dict(),

                        "cfg": dict(d=128, vocab_size=vocab_size, n_cells=4,

                                    n_organelles=8, n_memory=32, topk_organelle=4,

                                    topk_cell=2, L=4, seq_len=SEQ_LEN,

                                    batch_size=BATCH_SIZE, epochs=epochs, seed=SEED,

                                    epoch=epoch),

                        "epoch": epoch}, ckpt_epoch_path)

            print(f"[S5] CKPT saved (epoch {epoch}) -> {ckpt_epoch_path}", flush=True)

            if epoch == 1 and epochs == EPOCHS and e_sec > EPOCH_BUDGET_SEC:

                print(f"[S5] epoch1 耗时 {e_sec:.0f}s > 预算 {EPOCH_BUDGET_SEC}s，"

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

    comp = t_data_all + t_fwd_all + t_opt_all

    data_frac = t_data_all / comp if comp > 0 else 0.0

    print(f"[S5 {tag}] FINAL val_ppl={val_ppl:.4f} avg_gpu_util={avg_util:.1f}% "

          f"total={total_sec}s data_frac={data_frac * 100:.1f}%")

    return (val_ce, val_ppl, avg_util, total_sec, epoch_secs, data_frac)





def main():

    set_seed(SEED)

    vocab_size, train_ids, val_ids, chars = load_tinystories()

    print(f"[S5] tinystories 25MB(use_mb=25, 5ep完整) vocab={vocab_size} "

          f"train={train_ids.numel()} val={val_ids.numel()}")



    cell = FastHierLM(d=128, vocab_size=vocab_size, n_cells=4, n_organelles=8,

                      n_memory=32, topk_organelle=4, topk_cell=2, L=4,

                      layer_biases=LAYER_BIASES).to(DEVICE)

    n_cell_params = count_params(cell)

    print(f"[S5] CellMoE params={n_cell_params} (Fixed 延后一轮再跑)")



    ce, cp, cu, ct, es, df = train_one(cell, train_ids, val_ids, vocab_size,

                                       "CellMoE", True)

    # ---- 路由调制监控：mem_gate/assembly ratio（验收指标之二）----

    ratios = []

    for layer in cell.layers:

        for c in layer.cells:

            if getattr(c, '_ratios', None):

                ratios.extend(c._ratios)

    if ratios:

        last_vals = []

        for layer in cell.layers:

            for c in layer.cells:

                last_vals.append(round(c._ratios[-1], 4) if c._ratios else 0.0)

        print(f"[S5] mem_gate/assembly ratio mean={sum(ratios) / len(ratios):.4f} "

              f"(samples={len(ratios)}) last_per_cell={last_vals}", flush=True)

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

    torch.save({"state_dict": cell.state_dict(),

                "cfg": dict(d=128, vocab_size=vocab_size, n_cells=4,

                            n_organelles=8, n_memory=32, topk_organelle=4,

                            topk_cell=2, L=4, seq_len=SEQ_LEN,

                            batch_size=BATCH_SIZE, epochs=EPOCHS, seed=SEED),

                "final_ce": ce, "final_ppl": cp}, CKPT_PATH)

    print(f"[S5] CKPT saved -> {CKPT_PATH}")

    del cell

    if DEVICE == "cuda":

        torch.cuda.empty_cache()

    gc.collect()



    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    with open(OUT_CSV, "w", newline="") as f:

        w = csv.writer(f)

        w.writerow(["model", "params", "val_ce", "val_ppl", "gpu_util_pct",

                    "sec", "epoch_secs", "data_frac", "note"])

        w.writerow(["CellMoE", n_cell_params,

                    round(ce, 4), round(cp, 4), round(cu, 1), ct,

                    ";".join(map(str, es)), round(df, 4),

                    "Fixed 对比下一轮再跑"])

    print(f"[S5] CSV -> {OUT_CSV}")

    print(f"[S5] CellMoE PPL={cp:.4f}（Fixed 对比延后，本轮先出单模型结果）")





if __name__ == "__main__":

    main()

