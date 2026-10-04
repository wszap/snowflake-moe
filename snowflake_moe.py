# -*- coding: utf-8 -*-
"""Snowflake MoE —— 细胞化架构（CellMoE）最小版本

核心思想：
- 细胞器（Organelle）：20 个极简基础算子，单独存在无能力，靠"组装"产生能力。
- 忆点（MemoryPoint）：64 个条目，每项存 key 向量 + 20 维组装概率分布
  （表示"这类输入最常用哪几种拼法"）。
- 核架（CellFrame）：输入经忆点检索得到组装概率，Top-K 选出细胞器拼出临时结构。

验证目标：组合爆炸 + 概率组装能否 work（对比 fixed MoE 的固定专家）。

独立模块：不依赖 marvis_moe.py / marvis_moe_v7.py，可单独运行。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Organelle(nn.Module):
    """最小颗粒：极简基础算子，单独存在无能力。"""
    def __init__(self, d, h=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h, bias=False),
            nn.SiLU(),
            nn.Linear(h, d, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class AssemblyHead(nn.Module):
    """非线性组装头：残差 + 零初始化。

    assembly = x + fc2(SiLU(fc1(x)))
    初始时 fc2.weight=0, fc2.bias=0 → 输出恒等于 x，行为与纯线性组装完全一致；
    训练只学"残差"，不破坏线性先验。
    """
    def __init__(self, n_organelles, hidden=40):
        super().__init__()
        self.fc1 = nn.Linear(n_organelles, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, n_organelles)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


class SnowflakeMoE(nn.Module):
    """细胞化 MoE：n_organelles 细胞器 + n_memory 忆点 + 概率组装。"""
    def __init__(self, d, n_organelles=20, n_memory=64, topk_organelle=4):
        super().__init__()
        self.d = d
        self.n_organelles = n_organelles
        self.n_memory = n_memory
        self.topk = topk_organelle
        # 细胞器池
        self.organelles = nn.ModuleList([
            Organelle(d) for _ in range(n_organelles)
        ])
        # 忆点：key 向量 + 组装概率分布
        self.memory_keys = nn.Parameter(torch.randn(n_memory, d) * 0.1)
        self.memory_assembly = nn.Parameter(torch.zeros(n_memory, n_organelles))
        # 机制修正：组装标码非线性化（残差 + 零初始化，初始恒等=纯线性）
        self.assembly_head = AssemblyHead(n_organelles)
        # 输入编码
        self.encoder = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, d, bias=False)

    def forward(self, x):
        # x: [B, d]
        B = x.shape[0]
        v = self.norm(self.encoder(x))                    # [B, d]
        # 1. 检索忆点（向量相似度；训练用 Gumbel-Softmax 可导软选路，防坍缩）
        sim = v @ self.memory_keys.T / (self.d ** 0.5)    # [B, n_memory]
        if self.training:
            attn = F.gumbel_softmax(sim, tau=1.0, hard=False, dim=-1)
        else:
            attn = F.softmax(sim / 1.0, dim=-1)           # [B, n_memory]
        # 2. 取出组装标码（概率统计加权）→ 非线性组装头
        assembly_raw = attn @ self.memory_assembly       # [B, n_organelles]
        assembly = self.assembly_head(assembly_raw)      # [B, n_organelles]
        # 3. 概率化 + Top-K 稀疏
        weights = F.softmax(assembly, dim=-1)             # [B, n_organelles]
        topk_w, topk_idx = torch.topk(weights, self.topk, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)    # 重归一化
        # 4. 拼核架：只激活 top-K 细胞器
        # demo 版（double loop）已冒烟验证正确；训练用批量 gather 等价实现
        org_out = torch.stack([o(x) for o in self.organelles], dim=1)  # [B, n_org, d]
        w3 = topk_w.unsqueeze(-1)                                       # [B, topk, 1]
        idx3 = topk_idx.unsqueeze(-1).expand(-1, -1, self.d)            # [B, topk, d]
        out = (org_out.gather(1, idx3) * w3).sum(1)                     # [B, d]
        out = self.head(out)
        return out, {
            'weights': weights,          # 组装权重（监控用）
            'memory_attn': attn,         # 忆点检索分布
            'topk_idx': topk_idx,        # 实际激活的细胞器
        }


class SnowflakeMoE_LM(nn.Module):
    """字符级 LM 版 Snowflake：Embedding + in_proj + L×SnowflakeMoE + Linear head。

    与 train_lm.MarvisMoE_LM 同接口（forward 返回 (logits, aux/info)），
    便于直接复用莎士比亚数据管线与 eval_ppl。
    """
    def __init__(self, d, vocab_size, n_organelles=20, n_memory=64,
                 topk_organelle=4, L=2):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.L = L
        self.n_organelles = n_organelles
        self.n_memory = n_memory
        self.topk = topk_organelle
        self.embed = nn.Embedding(vocab_size, d)
        self.in_proj = nn.Linear(d, d, bias=False)
        self.layers = nn.ModuleList([
            SnowflakeMoE(d, n_organelles=n_organelles, n_memory=n_memory,
                         topk_organelle=topk_organelle)
            for _ in range(L)
        ])
        self.head = nn.Linear(d, vocab_size)

    def forward(self, tokens):
        x = self.embed(tokens)                            # [B, T, d]
        x = self.in_proj(x)
        info = None
        for layer in self.layers:
            B, T, d = x.shape
            y, li = layer(x.reshape(B * T, d))
            info = li if info is None else info
            x = y.reshape(B, T, d)
        return self.head(x), info                          # (B, T, vocab)


class CellMoE(nn.Module):
    """v3 单级细胞化 MoE：纯线性组装（无组装头）。

    与 SnowflakeMoE(00005 残差头) 区分：本类保持 00003 的纯线性结构
    assembly = attn @ memory_assembly，作为两级架构的第一级子细胞，
    不引入非线性组装头（验证目标仅为"层级嵌套"，一次只改一个机制）。
    """
    def __init__(self, d, n_organelles=8, n_memory=32, topk_organelle=4):
        super().__init__()
        self.d = d
        self.n_organelles = n_organelles
        self.n_memory = n_memory
        self.topk = topk_organelle
        self.organelles = nn.ModuleList([
            Organelle(d) for _ in range(n_organelles)
        ])
        self.memory_keys = nn.Parameter(torch.randn(n_memory, d) * 0.1)
        self.memory_assembly = nn.Parameter(torch.zeros(n_memory, n_organelles))
        # 终身学习扩展槽：新增忆点（默认 None 不启用；启用时 forward 自动拼接）
        self.new_memory_keys = None
        self.new_memory_assembly = None
        self.encoder = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, d, bias=False)

    def add_new_memory(self, add=8, seed=2026):
        """扩展忆点库：追加 add 个新忆点（key 随机×0.1, assembly 全零）。"""
        g = torch.Generator(device=self.memory_keys.device).manual_seed(seed)
        d = self.d
        dev = self.memory_keys.device
        new_keys = torch.randn(add, d, generator=g, device=dev) * 0.1
        new_asm = torch.zeros(add, self.n_organelles, device=dev)
        if self.new_memory_keys is None:
            self.new_memory_keys = nn.Parameter(new_keys)
            self.new_memory_assembly = nn.Parameter(new_asm)
        else:
            self.new_memory_keys = nn.Parameter(
                torch.cat([self.new_memory_keys.detach(), new_keys], dim=0))
            self.new_memory_assembly = nn.Parameter(
                torch.cat([self.new_memory_assembly.detach(), new_asm], dim=0))

    def forward(self, x):
        # x: [B, d]
        v = self.norm(self.encoder(x))                    # [B, d]
        # 1. 检索忆点（Gumbel-Softmax 可导软选路，防坍缩）
        if self.new_memory_keys is None:
            keys = self.memory_keys
            asm = self.memory_assembly
        else:
            keys = torch.cat([self.memory_keys, self.new_memory_keys], dim=0)
            asm = torch.cat([self.memory_assembly, self.new_memory_assembly], dim=0)
        sim = v @ keys.T / (self.d ** 0.5)                # [B, n_memory(+add)]
        if self.training:
            attn = F.gumbel_softmax(sim, tau=1.0, hard=False, dim=-1)
        else:
            attn = F.softmax(sim / 1.0, dim=-1)           # [B, n_memory(+add)]
        # 2. 纯线性组装标码（v3 结构，无组装头）
        assembly = attn @ asm                              # [B, n_organelles]
        # 3. 概率化 + Top-K 稀疏
        weights = F.softmax(assembly, dim=-1)
        topk_w, topk_idx = torch.topk(weights, self.topk, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)
        # 4. 拼核架：只激活 top-K 细胞器
        org_out = torch.stack([o(x) for o in self.organelles], dim=1)  # [B, n_org, d]
        w3 = topk_w.unsqueeze(-1)
        idx3 = topk_idx.unsqueeze(-1).expand(-1, -1, self.d)
        out = (org_out.gather(1, idx3) * w3).sum(1)                     # [B, d]
        out = self.head(out)
        return out, {
            'weights': weights,
            'memory_attn': attn,
            'topk_idx': topk_idx,
        }


class HierarchicalCellMoE(nn.Module):
    """两级细胞化架构：n_cells 个 CellMoE（细胞层）+ 上层路由（组织层）。

    第一级：每个细胞是完整 v3 CellMoE（规模缩小：8 细胞器 / 32 忆点）。
    第二级：输入编码后经 top_router 打分，Gumbel-Softmax 软路由加权聚合。
    验证核心问题：两级嵌套（细胞 → 组织）是否有性能增益。
    """
    def __init__(self, d, n_cells=4, n_organelles=8, n_memory=32,
                 topk_organelle=4, topk_cell=2):
        super().__init__()
        self.d = d
        self.n_cells = n_cells
        self.topk_cell = topk_cell
        # 第一级：n_cells 个独立 CellMoE（v3 完整结构，规模缩小）
        self.cells = nn.ModuleList([
            CellMoE(d, n_organelles=n_organelles, n_memory=n_memory,
                    topk_organelle=topk_organelle)
            for _ in range(n_cells)
        ])
        # 第二级：上层路由（在细胞之间选择）
        self.encoder = nn.Linear(d, d, bias=False)
        self.top_router = nn.Linear(d, n_cells, bias=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        # 第一级：每个细胞独立计算
        cell_outs = []
        cell_infos = []
        for cell in self.cells:
            out, info = cell(x)
            cell_outs.append(out)         # [B, d]
            cell_infos.append(info)
        cell_outs = torch.stack(cell_outs, dim=1)   # [B, n_cells, d]

        # 第二级：上层路由（softmax，用于聚合）
        v = self.norm(self.encoder(x))                # [B, d]
        gate_logits = self.top_router(v)              # [B, n_cells]

        # Gumbel-Softmax 保持可导（沿用 v3 的修复思路）
        if self.training:
            gate = F.gumbel_softmax(gate_logits, tau=1.0, hard=False, dim=-1)
        else:
            gate = F.softmax(gate_logits, dim=-1)

        # 加权聚合
        out = (gate.unsqueeze(-1) * cell_outs).sum(dim=1)   # [B, d]

        info = {
            'gate': gate,                    # [B, n_cells]
            'cell_infos': cell_infos,
        }
        return out, info


class HierarchicalCellMoE_LM(nn.Module):
    """字符级 LM 版两级细胞化：Embedding + in_proj + L×HierarchicalCellMoE + head。

    与 train_lm.MarvisMoE_LM 同接口（forward 返回 (logits, info)），
    复用莎士比亚数据管线与 eval_ppl。
    """
    def __init__(self, d, vocab_size, n_cells=4, n_organelles=8, n_memory=32,
                 topk_organelle=4, topk_cell=2, L=2):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.L = L
        self.n_cells = n_cells
        self.n_organelles = n_organelles
        self.n_memory = n_memory
        self.topk = topk_organelle
        self.embed = nn.Embedding(vocab_size, d)
        self.in_proj = nn.Linear(d, d, bias=False)
        self.layers = nn.ModuleList([
            HierarchicalCellMoE(d, n_cells=n_cells, n_organelles=n_organelles,
                                n_memory=n_memory, topk_organelle=topk_organelle,
                                topk_cell=topk_cell)
            for _ in range(L)
        ])
        self.head = nn.Linear(d, vocab_size)

    def forward(self, tokens):
        x = self.embed(tokens)                            # [B, T, d]
        x = self.in_proj(x)
        info = None
        for layer in self.layers:
            B, T, d = x.shape
            y, li = layer(x.reshape(B * T, d))
            info = li if info is None else info
            x = y.reshape(B, T, d)
        return self.head(x), info                          # (B, T, vocab)
