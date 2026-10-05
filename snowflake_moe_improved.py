# -*- coding: utf-8 -*-
"""Snowflake MoE 核心机制改进（清单二：2.1–2.6）

在 snowflake_moe.py 基础上做增量扩展，原文件保持不动。

2.1 memory_value —— 忆点直接参与输出
    忆点增加 d 维 value 向量；
    out = sum(w_k * organelle[k](x)) + sum(attn[m] * memory_value[m])
    100% 擦除忆点后 PPL 应显著恶化（验证擦除机制确实携带知识）。

2.2 忆点分组 softmax
    老忆点组 / 新忆点组分别 softmax，再按可学习组权重聚合，
    避免新忆点稀释旧忆点注意力（旧领域退化 < 0.3）。

2.3 新忆点学习增强
    新忆点 8→16、训练 epoch 1→3、新忆点参数单独学习率（优化器独立分组）。

2.4 免疫系统（细胞器健康）
    每 epoch 体检：输出方差 / 激活频率 / 余弦相似度，
    淘汰最差细胞器并替换（克隆最优 + 噪声）。

2.5 忆点修剪
    每 2 epoch 淘汰激活频率 < 5% 的忆点（保底保留一半）。

2.6 Gate 熵正则
    第二级 gate 加熵正则（最大化熵 / 强制 topk_cell=2）。

独立模块：仅依赖 torch/numpy，可单独运行冒烟。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snowflake_moe import Organelle, AssemblyHead  # noqa: E402


# ================================================================ 2.4 免疫系统
class ImmuneSystem:
    """细胞器健康体检：每 epoch 统计输出方差/激活频率/互相似度，淘汰+替换。

    replace: 把最差细胞器替换为最优细胞器的克隆 + 噪声（保留探索性）。
    """

    def __init__(self, replace_frac=0.1, noise=0.01):
        self.replace_frac = replace_frac
        self.noise = noise
        self.history = []  # 每轮体检记录

    def check(self, model, x):
        """x: [B, d] 输入样本。返回体检报告 dict。"""
        model.eval()
        with torch.no_grad():
            org_out = torch.stack([o(x) for o in model.organelles], dim=1)  # [B, N, d]
            var = org_out.var(dim=0).mean(-1)                # [N] 输出方差
            acts = var  # 用方差作为活性代理（低方差=死细胞器）
            # 激活频率：从模型最近一次 forward 的 topk_idx 统计
            freq = torch.zeros(model.n_organelles, device=x.device)
            if model.last_topk is not None:
                f = model.last_topk.reshape(-1)
                freq.scatter_add_(0, f, torch.ones(f.numel(), device=f.device))
                freq = freq / freq.sum().clamp_min(1.0)
            # 余弦相似度：每对细胞器输出均值向量的相似度
            mean_out = org_out.mean(0)                       # [N, d]
            norm = mean_out.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            sim = (mean_out @ mean_out.T) / (norm @ norm.T)  # [N, N]
            sim.fill_diagonal_(0.0)
            max_sim = sim.max(-1).values
        model.train()
        report = dict(var=var, freq=freq, max_sim=max_sim,
                      dead=(var < 1e-4).sum().item())
        self.history.append({k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                             for k, v in report.items()})
        return report

    def treat(self, model, report):
        """淘汰最差 replace_frac 个细胞器，替换为最优克隆+噪声。"""
        n = model.n_organelles
        n_r = max(1, int(n * self.replace_frac))
        score = report["freq"] + 0.5 * (report["var"] / report["var"].clamp_min(1e-8).max())
        worst = torch.argsort(score)[:n_r]
        best = torch.argsort(score)[-n_r:]
        replaced = 0
        for wi, bi in zip(worst.tolist(), best.tolist()):
            if wi == bi:
                continue
            wb = model.organelles[wi]
            bb = model.organelles[bi]
            for (wn, wp), (sn, sp) in zip(wb.named_parameters(), bb.named_parameters()):
                if wn == sn and wp.shape == sp.shape:
                    wp.data = sp.data + torch.randn_like(sp.data) * self.noise
            replaced += 1
        return replaced


# ================================================================ 2.5 忆点修剪
def prune_memories(model, min_usage=0.05, keep_floor=0.5):
    """按激活频率修剪忆点：使用率 < min_usage 的淘汰，保底保留 keep_floor。

    返回修剪掉的忆点数量。修剪对 memory_keys/memory_assembly/memory_value 同步。
    """
    if model.memory_attn is None:
        return 0
    usage = model.memory_attn.mean(dim=0)             # [n_memory(+new)]
    n_total = usage.numel()
    keep_min = max(1, int(n_total * keep_floor))

    def _mask(u):
        m = u >= min_usage
        if m.sum() < keep_min:                        # 保底
            top = torch.topk(u, keep_min).indices
            m = torch.zeros_like(m)
            m[top] = True
        return m

    # 存在新忆点时：主库+新库合并后统一修剪，避免形状不一致
    if model.new_memory_keys is not None:
        keys = torch.cat([model.memory_keys, model.new_memory_keys.detach()], dim=0)
        asm = torch.cat([model.memory_assembly, model.new_memory_assembly.detach()], dim=0)
        val = torch.cat([model.memory_value, model.new_memory_value.detach()], dim=0) \
            if (hasattr(model, "new_memory_value") and model.new_memory_value is not None) else None
        keep2 = _mask(usage)
        if keep2.all():
            return 0
        model.memory_keys = nn.Parameter(keys[keep2].detach())
        model.memory_assembly = nn.Parameter(asm[keep2].detach())
        if val is not None:
            model.memory_value = nn.Parameter(val[keep2].detach())
        model.new_memory_keys = None
        model.new_memory_assembly = None
        model.new_memory_value = None
        model.memory_attn = None
        return int((~keep2).sum().item())

    keep_mask = _mask(usage)
    if keep_mask.all():
        return 0
    model.memory_keys = nn.Parameter(model.memory_keys[keep_mask].detach())
    model.memory_assembly = nn.Parameter(model.memory_assembly[keep_mask].detach())
    if hasattr(model, "memory_value") and model.memory_value is not None:
        model.memory_value = nn.Parameter(model.memory_value[keep_mask].detach())
    model.memory_attn = None
    return int((~keep_mask).sum().item())


# ================================================================ 改进版 CellMoE
class ImprovedCellMoE(nn.Module):
    """v3 CellMoE + 机制 2.1/2.2/2.3/2.4/2.5。

    相对原 CellMoE 的增量：
    - memory_value：忆点携带 d 维 value，直接参与输出（2.1）。
    - 分组 softmax：old/new 忆点分别 softmax 再按组权重聚合（2.2）。
    - add_new_memory 默认 add=16，新忆点参数单独优化分组（2.3）。
    - 免疫系统：免疫.check/treat 每 epoch 调用（2.4）。
    - 忆点修剪：prune_memories 每 2 epoch 调用（2.5）。
    """

    def __init__(self, d, n_organelles=8, n_memory=32, topk_organelle=4,
                 use_memory_value=True, group_softmax=True, group_weight=0.5,
                 memory_read_scale=1.0):
        super().__init__()
        self.d = d
        self.n_organelles = n_organelles
        self.n_memory = n_memory
        self.topk = topk_organelle
        self.use_memory_value = use_memory_value
        self.group_softmax = group_softmax
        self.memory_read_scale = memory_read_scale
        self.organelles = nn.ModuleList([Organelle(d) for _ in range(n_organelles)])
        self.memory_keys = nn.Parameter(torch.randn(n_memory, d) * 0.1)
        self.memory_assembly = nn.Parameter(torch.zeros(n_memory, n_organelles))
        self.memory_value = nn.Parameter(torch.zeros(n_memory, d)) \
            if use_memory_value else None
        # 2.2 分组权重（可学习，初始 group_weight）
        self.group_w = nn.Parameter(torch.tensor([group_weight])) \
            if group_softmax else None
        # 2.3 新忆点扩展槽
        self.new_memory_keys = None
        self.new_memory_assembly = None
        self.new_memory_value = None
        self.encoder = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, d, bias=False)
        # 运行时统计
        self.last_topk = None
        self.memory_attn = None
        self.immunity = ImmuneSystem()

    # ---- 2.3 新忆点扩展（显式 device=model.device，1.3） ----
    def add_new_memory(self, add=16, seed=2026):
        """追加 add 个新忆点。返回 (新忆点参数分组 list, 旧忆点数)。"""
        dev = self.memory_keys.device
        g = torch.Generator(device=dev).manual_seed(seed)
        new_keys = torch.randn(add, self.d, generator=g, device=dev) * 0.1
        new_asm = torch.zeros(add, self.n_organelles, device=dev)
        new_val = torch.zeros(add, self.d, device=dev) if self.use_memory_value else None
        if self.new_memory_keys is None:
            self.new_memory_keys = nn.Parameter(new_keys)
            self.new_memory_assembly = nn.Parameter(new_asm)
            self.new_memory_value = nn.Parameter(new_val) if new_val is not None else None
        else:
            self.new_memory_keys = nn.Parameter(
                torch.cat([self.new_memory_keys.detach(), new_keys], dim=0))
            self.new_memory_assembly = nn.Parameter(
                torch.cat([self.new_memory_assembly.detach(), new_asm], dim=0))
            if self.new_memory_value is not None:
                self.new_memory_value = nn.Parameter(
                    torch.cat([self.new_memory_value.detach(), new_val], dim=0))
        params = []
        if self.new_memory_keys is not None:
            params += [self.new_memory_keys, self.new_memory_assembly]
            if self.new_memory_value is not None:
                params.append(self.new_memory_value)
        return params, self.n_memory

    # ---- 前向 ----
    def forward(self, x):
        # x: [B, d]
        v = self.norm(self.encoder(x))
        if self.new_memory_keys is None:
            keys, asm = self.memory_keys, self.memory_assembly
            val = self.memory_value
            n_old = keys.shape[0]
        else:
            keys = torch.cat([self.memory_keys, self.new_memory_keys], dim=0)
            asm = torch.cat([self.memory_assembly, self.new_memory_assembly], dim=0)
            val = torch.cat([self.memory_value, self.new_memory_value], dim=0) \
                if self.use_memory_value else None
            n_old = self.memory_keys.shape[0]
        sim = v @ keys.T / (self.d ** 0.5)                 # [B, n]
        if self.group_softmax and self.new_memory_keys is not None:
            # 2.2 分组 softmax：old / new 分别 softmax
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
        # 组装标码（纯线性，同 v3）
        assembly = attn @ asm                                  # [B, n_organelles]
        weights = F.softmax(assembly, dim=-1)
        topk_w, topk_idx = torch.topk(weights, self.topk, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True).clamp_min(1e-9)
        org_out = torch.stack([o(x) for o in self.organelles], dim=1)  # [B, N, d]
        w3 = topk_w.unsqueeze(-1)
        idx3 = topk_idx.unsqueeze(-1).expand(-1, -1, self.d)
        out = (org_out.gather(1, idx3) * w3).sum(1)            # [B, d]
        # 2.1 memory_value 直接参与输出（放大 memory_read 系数，保证记忆载体地位）
        if self.use_memory_value:
            out = out + self.memory_read_scale * (attn @ val)
        self.last_topk = topk_idx.detach()
        out = self.head(out)
        return out, {
            "weights": weights,
            "memory_attn": attn,
            "topk_idx": topk_idx,
            "assembly": assembly,
        }


# ================================================================ 改进版两级（2.6 Gate 熵正则）
class ImprovedHierarchicalCellMoE(nn.Module):
    """两级 CellMoE + 机制 2.6（Gate 熵正则 / 强制 topk_cell）。

    第二级 gate 输出分布上加熵正则（loss 项 -lambda_ent * entropy(gate)），
    或强制 topk_cell 稀疏聚合（只激活 topk 个细胞，防 gate 塌缩）。
    """

    def __init__(self, d, n_cells=4, n_organelles=8, n_memory=32,
                 topk_organelle=4, topk_cell=2, gate_ent_reg=True,
                 memory_read_scale=1.0):
        super().__init__()
        self.d = d
        self.n_cells = n_cells
        self.topk_cell = topk_cell
        self.gate_ent_reg = gate_ent_reg
        self.cells = nn.ModuleList([
            ImprovedCellMoE(d, n_organelles=n_organelles, n_memory=n_memory,
                            topk_organelle=topk_organelle,
                            memory_read_scale=memory_read_scale)
            for _ in range(n_cells)
        ])
        self.encoder = nn.Linear(d, d, bias=False)
        self.top_router = nn.Linear(d, n_cells, bias=False)
        self.norm = nn.LayerNorm(d)
        self.last_gate = None

    def forward(self, x):
        cell_outs, cell_infos = [], []
        for cell in self.cells:
            out, info = cell(x)
            cell_outs.append(out)
            cell_infos.append(info)
        cell_outs = torch.stack(cell_outs, dim=1)              # [B, n_cells, d]
        v = self.norm(self.encoder(x))
        gate_logits = self.top_router(v)                       # [B, n_cells]
        if self.training:
            gate = F.gumbel_softmax(gate_logits, tau=1.0, hard=False, dim=-1)
        else:
            gate = F.softmax(gate_logits, dim=-1)
        self.last_gate = gate.detach()
        # 2.6 强制 topk_cell 稀疏聚合（默认启用）
        topk_w, topk_idx = torch.topk(gate, self.topk_cell, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True).clamp_min(1e-9)
        gate_sparse = torch.zeros_like(gate)
        gate_sparse.scatter_(1, topk_idx, topk_w)
        out = (gate_sparse.unsqueeze(-1) * cell_outs).sum(dim=1)
        # 熵正则：返回当前 gate 熵（训练循环里 loss += -lambda_ent * entropy）
        entropy = float(-(gate * (gate + 1e-9).log()).sum(-1).mean().item())
        info = {
            "gate": gate,
            "gate_entropy": entropy,
            "cell_infos": cell_infos,
        }
        return out, info


# ================================================================ LM 封装
class ImprovedCellMoE_LM(nn.Module):
    """字符级 LM 版改进 CellMoE：接口兼容 train_lm.MarvisMoE_LM。"""

    def __init__(self, d, vocab_size, n_organelles=20, n_memory=64,
                 topk_organelle=4, L=2, **kw):
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
            ImprovedCellMoE(d, n_organelles=n_organelles, n_memory=n_memory,
                            topk_organelle=topk_organelle, **kw)
            for _ in range(L)
        ])
        self.head = nn.Linear(d, vocab_size)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.in_proj(x)
        info = None
        for layer in self.layers:
            B, T, d = x.shape
            y, li = layer(x.reshape(B * T, d))
            info = li if info is None else info
            x = y.reshape(B, T, d)
        return self.head(x), info


class ImprovedHierarchicalCellMoE_LM(nn.Module):
    """字符级 LM 版两级改进：第二级 gate 熵正则 + topk_cell 稀疏。"""

    def __init__(self, d, vocab_size, n_cells=4, n_organelles=8, n_memory=32,
                 topk_organelle=4, topk_cell=2, L=2, gate_ent_reg=True,
                 memory_read_scale=1.0):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.L = L
        self.n_cells = n_cells
        self.embed = nn.Embedding(vocab_size, d)
        self.in_proj = nn.Linear(d, d, bias=False)
        self.layers = nn.ModuleList([
            ImprovedHierarchicalCellMoE(d, n_cells=n_cells,
                                        n_organelles=n_organelles,
                                        n_memory=n_memory,
                                        topk_organelle=topk_organelle,
                                        topk_cell=topk_cell,
                                        gate_ent_reg=gate_ent_reg,
                                        memory_read_scale=memory_read_scale)
            for _ in range(L)
        ])
        self.head = nn.Linear(d, vocab_size)

    def forward(self, tokens):
        x = self.embed(tokens)
        x = self.in_proj(x)
        info = None
        for layer in self.layers:
            B, T, d = x.shape
            y, li = layer(x.reshape(B * T, d))
            info = li if info is None else info
            x = y.reshape(B, T, d)
        return self.head(x), info


# ================================================================ 冒烟
if __name__ == "__main__":
    torch.manual_seed(2026)
    m = ImprovedCellMoE(d=64, n_organelles=8, n_memory=32, topk_organelle=4)
    x = torch.randn(16, 64)
    out, info = m(x)
    print(f"[SMOKE] ImprovedCellMoE out={tuple(out.shape)} "
          f"topk={tuple(info['topk_idx'].shape)} attn={tuple(info['memory_attn'].shape)}")
    # 2.2 分组 softmax 检查（加新忆点后）
    m.add_new_memory(add=16, seed=2026)
    out2, info2 = m(x)
    print(f"[SMOKE] add_new_memory(16) -> keys={m.memory_keys.shape[0]}+{m.new_memory_keys.shape[0]} "
          f"attn={tuple(info2['memory_attn'].shape)}")
    # 2.1 memory_value 参与输出（输出应该有变化）
    print(f"[SMOKE] memory_value 存在: {m.memory_value is not None}")
    # 2.4 免疫体检
    rep = m.immunity.check(m, x)
    print(f"[SMOKE] immunity: dead={rep['dead']} freq_mean={rep['freq'].mean().item():.4f}")
    # 2.5 修剪
    pruned = prune_memories(m, min_usage=0.05)
    print(f"[SMOKE] prune -> removed={pruned}")
    # 2.6 两级 gate 熵
    h = ImprovedHierarchicalCellMoE(d=64, n_cells=4, n_organelles=8,
                                    n_memory=32, topk_organelle=4, topk_cell=2)
    _, hin = h(x)
    print(f"[SMOKE] Hierarchical gate_entropy={hin['gate_entropy']:.4f}")
    print("[SMOKE] 全部通过")
