# -*- coding: utf-8 -*-
"""paperplane-Moe V1.001 · 生物学观测层 + 残差版真拼 训练入口（零改动现有源码）。

本文件复用 train_compose.py / train_split.py 的 build、make_clip、do_split、
train_v3 的数据与 evaluate，只新增三件事：

  1. 【观测层真正落地】规格 5/11/13 的字段不再只写在文档里：
     divide / wire / cell_call / temp_module / temp_expert(+dissolve) /
     mem_kind / budget / transport / health 全部写进 events。
  2. 【残差版真拼】--cell-type residual → ResidualComposeCell：
     out = B 主干 + gate·Σ w_i·silu(xE1_i)·E2_i，gate=0 严格恒等，
     修 step25 的 +0.1099（换起步函数类会输）。
  3. 【生物学原语状态机】受体 / 血液两类忆点 / 小核架 / 稀疏回顾 /
     临时专家模组 / ATP / 自噬 / 主动死亡 / 酶 / 局部梯度防护 /
     线粒体 / 叶绿体，全部是独立旁路对象（不挂到 model.parameters()），
     默认 enable=False ⇒ 前向逐位恒等、不消耗全局 RNG、不改 optimizer。

⚠ 诚实边界：原语的前向【注入】接口已写好（gain>0 时返回调制张量），
   但 model.forward 把嵌入封装在内部，V1.001 只把「临时专家模组的正锥 Δw」
   通过 cell.temp_delta_w 真正接进前向；其余原语是旁路观测 + 状态机。
   全量前向注入排在 V1.002（需 forward hook 或改 snowflake_v4.forward）。

用法（在 v2_compositional/ 下）：
  # ① 零污染自检：默认 B 档 + 观测层，应逐位复现 b5_E1_split
  python3 train_bio.py --data data_25mb.txt --cell-type B --seed 0 \
      --max-steps 3000 --tag bio_s0_B --zero-check
  # ② 残差版真拼（同 seed 配对）
  python3 train_bio.py --data data_25mb.txt --cell-type residual --seed 0 \
      --max-steps 3000 --tag bio_s0_residual
  # ③ 打开全部生物学原语（需先做完 ① 的零污染自检）
  python3 train_bio.py --data data_25mb.txt --cell-type residual --seed 0 \
      --max-steps 3000 --bio-on --tag bio_s0_full
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

# ---- 让 src/ 与 v2_compositional/ 都在 sys.path（从任一目录运行都成立）----
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from paperplane_moe import (
    BioConfig, Bloodstream, Receptor, Nucleoskeleton, SparseRecall,
    TempExpertModule, ATPBudget, Enzyme, GradientGuard, Mitochondria,
    Chloroplast, Autophagy, Apoptosis, TelemetryRecorder,
    ResidualComposeCell, param_overhead, replay_lineage,
)
from paperplane_moe.telemetry import validate_events, CORE_EVENT_KINDS
from paperplane_moe.enzymology import Enzymology, ENZ_EVENT_KINDS
from paperplane_moe.enzyme_rate import EnzymeRateController   # V1.004
from bio_update import BioUpdate
from train_v3 import (load_data, evaluate, resolve_device,
                      configure_runtime)
from train_compose import all_cells, make_clip, do_split_v10, _make_v10_cls
from train_split import do_split as do_split_B
from organelle_registry import OrganelleRegistry
from organelle_groups import SoftGroups
import snowflake_v4
from snowflake_v4 import SnowflakeLMV4

_ORIG_B = snowflake_v4.SnowflakeB


# ---------------------------------------------------------------- 构建
def _make_residual_cls(a, cfg):
    class _ResidualCell(ResidualComposeCell):
        def __init__(self, **kw):
            kw.update(k=a.compose_k, h_cell=a.h_cell,
                      mode=a.compose_mode, bio_cfg=cfg,
                      seed=int(kw.get("seed", 0)))
            super().__init__(**kw)
    return _ResidualCell


def build_bio(a, cfg, vocab, n_start, device):
    if a.cell_type == "B":
        snowflake_v4.SnowflakeB = _ORIG_B
    elif a.cell_type == "residual":
        snowflake_v4.SnowflakeB = _make_residual_cls(a, cfg)
    elif a.cell_type == "v10":
        snowflake_v4.SnowflakeB = _make_v10_cls(a)
    else:
        raise ValueError(a.cell_type)
    return SnowflakeLMV4(
        vocab_size=vocab, d=a.d, n=n_start, L=a.L, h=a.h,
        n_cells=a.n_cells, seq_len=a.seq_len, rank=a.rank, temp=a.temp,
        n_head=a.n_head, scaffold_top_k=a.scaffold_top_k,
        select_mode=a.select_mode, variant=a.variant,
        shared_mult=a.shared_mult, seed=a.seed,
        clone_temp=a.clone_temp, groups=0).to(device)


def cfg_from_args(a):
    """--bio-on 打开全部原语；默认全关（纯观测层，严格零扰动）。"""
    on = bool(a.bio_on)
    return BioConfig(
        enable=on, receptor=on, blood=on, active_transport=on,
        temp_expert=on, skeleton=on, sparse_recall=on,
        autophagy=on, apoptosis=on, atp=on, enzyme=on,
        gradient_guard=on, mitochondria=on, chloroplast=on,
        clone_in_eval=on,
        k=a.compose_k, h_cell=a.h_cell, top_m=4, skeleton_rank=4,
        temp_module_ttl=a.temp_module_ttl,
        budget_init=(1000.0 if on else 0.0),
        # ★ gate 单独由 --bio-gain 控制；默认 0 → 即使 --bio-on 也严格恒等
        gate=float(a.bio_gain), eps=1e-6,
    )


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--h", type=int, default=128)
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--scaffold-top-k", type=int, default=32)
    ap.add_argument("--select-mode", default="soft")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--temp", type=float, default=5.0)
    ap.add_argument("--variant", default="shared")
    ap.add_argument("--shared-mult", type=int, default=4)
    ap.add_argument("--clone-temp", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-k", type=int, default=8)
    ap.add_argument("--split-at", type=int, default=1500)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--join-mode", choices=["share", "compete"], default="share")
    ap.add_argument("--mode", choices=["split"], default="split")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default="output/v4")
    # ---- step25 兼容 ----
    ap.add_argument("--cell-type", choices=["B", "residual", "v10"], default="B")
    ap.add_argument("--h-cell", type=int, default=3)
    ap.add_argument("--compose-k", type=int, default=16)
    ap.add_argument("--compose-renorm", type=int, default=1)
    ap.add_argument("--compose-chunk", type=int, default=1)
    ap.add_argument("--compose-mode", choices=["fixed", "learned"], default="fixed")
    ap.add_argument("--mem-budget", type=float, default=256.0)
    ap.add_argument("--wiring-mod", choices=["none", "norm", "center"], default="none")
    ap.add_argument("--clip-mode", choices=["global", "grouped"], default="global")
    # ---- V1.001 新增 ----
    ap.add_argument("--bio-on", action="store_true",
                    help="打开全部生物学原语（默认纯观测层、严格零扰动）")
    ap.add_argument("--bio-gain", type=float, default=0.0,
                    help="所有原语的统一调制增益；0=严格恒等（默认）")
    ap.add_argument("--temp-module-ttl", type=int, default=0,
                    help="临时专家模组存活步数；0=同 step 创建并解散")
    ap.add_argument("--temp-module-every", type=int, default=500,
                    help="每隔多少步记录一次临时模组生命周期")
    ap.add_argument("--zero-check", action="store_true",
                    help="跑完与 st25_s0_B.json 做 301 点逐位对照")
    # ---- V1.004：酶 = per-cell 参数更新速率常数（不是乘前向）----
    ap.add_argument("--enz-rate-mode", default="off",
                    choices=["off", "amp", "cat", "cat_het", "const", "lrx"],
                    help="off=历史逐位 / amp=V1.003旧前向对照 / cat=主实验 / "
                         "cat_het=Km自主漂变 / const=全局k / lrx=lr×k̄")
    ap.add_argument("--enz-beta-const", type=float, default=0.10,
                    help="组成型管家酶（cell 出生即在）")
    ap.add_argument("--enz-beta-ind", type=float, default=4.0,
                    help="底物诱导酶强度")
    ap.add_argument("--enz-tau-e", type=float, default=60.0,
                    help="诱导/降解慢变量时间常数（Jacob–Monod）")
    ap.add_argument("--enz-k-max", type=float, default=6.0)
    ap.add_argument("--enz-gamma-km", type=float, default=0.03,
                    help="cat_het 的 Km 漂变速率")
    ap.add_argument("--enz-freeze-cell", type=int, default=-1,
                    help="速率 rescue：把该 cell 的 k 固定为 1（不删专家）")
    ap.add_argument("--enz-cost-alpha", type=float, default=0.10,
                    help="双线成本里酶浓度的记账权重")
    ap.add_argument("--enz-scope", default="gate", choices=["gate", "backbone"],
                    help="gate=step29 原作用面(16 个 gate 标量) / "
                         "backbone=V1.005：k_i 落 16 份主干 lr")
    ap.add_argument("--enz-warmup", type=int, default=1,
                    help="诱导延迟步数：前 N 步严格 k=1（Jacob-Monod 转录延迟）")
    # ---- V1.010：参数量归一的守恒资源分配（不改拼专家前向）----
    ap.add_argument("--bio-update", choices=["off", "on"], default="off",
                    help="on=在 shared+每个 compose cell 间守恒搬运更新预算")
    ap.add_argument("--bio-c-weight", type=float, default=0.0,
                    help="组间相对优化压力的实验权重；0 是严格基线")
    ap.add_argument("--bio-use-C", action="store_true",
                    help="兼容旧试验名；等价于 --bio-c-weight 1（若未显式设置）")
    ap.add_argument("--bio-trust", type=float, default=4.0,
                    help="相对基线学习率上限，必须 >=1；1 是冻结分配对照")
    ap.add_argument("--bio-tau-e", type=float, default=100.0,
                    help="每组梯度 RMS 的慢 EMA 时间常数")
    ap.add_argument("--bio-kappa", type=float, default=0.30,
                    help="资源份额回到参数量基线的对数回复强度")
    ap.add_argument("--bio-eta-phi", type=float, default=0.05,
                    help="负熵镜像下降步长，只作用于资源份额 phi")
    ap.add_argument("--bio-trace-flush-every", type=int, default=10,
                    help="phi JSONL 的刷新间隔；每一步仍写一行")
    ap.add_argument("--bio-require-active", action="store_true",
                    help="要求组间相对尺度确实分化，否则以 rc=7 失败")
    ap.add_argument("--bio-min-scale-std", type=float, default=0.005,
                    help="G-PHI 作用面硬闸的最小加权相对尺度标准差")
    # ---- 本地运行时：不改变默认实验语义 ----
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"],
                    default="auto")
    ap.add_argument("--cpu-threads", type=int, default=0,
                    help="CPU intra-op threads；本机小模型实测 2 最快")
    ap.add_argument("--interop-threads", type=int, default=0,
                    help="CPU inter-op threads；0 保持 PyTorch 默认")
    ap.add_argument("--token-cache", default="",
                    help="token 缓存路径，或 auto=DATA.tokens.pt")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false",
                    default=True)
    a = ap.parse_args()

    device = resolve_device(a.device)
    configure_runtime(device, a.seed, a.cpu_threads, a.interop_threads)
    bf16 = a.bf16 and device.type in ("cuda", "xpu")
    tag = a.tag or ("bio_s%d_%s" % (a.seed, a.cell_type))
    cfg = cfg_from_args(a)

    # ---- 生物学旁路对象（★ 全部不挂到 model，不进 parameters/optimizer）----
    tel = TelemetryRecorder(cfg, n_cells=a.L * a.n_cells, n_organelles=a.n)
    blood = Bloodstream(a.d, a.L, cfg, seed=a.seed)
    receptor = Receptor(a.d, cfg)
    skeleton = Nucleoskeleton(a.d, cfg)
    recall = SparseRecall(a.d, cfg)
    temp_mod = TempExpertModule(a.n, a.d, cfg)
    atp = ATPBudget(cfg)
    autophagy, apoptosis = Autophagy(cfg), Apoptosis(cfg)
    enzyme, chloroplast = Enzyme(a.d, cfg), Chloroplast(a.d, cfg, seed=a.seed)
    guard, mito = GradientGuard(cfg), Mitochondria(cfg)
    # ---- step28e device 对齐 ----
    # 旁路 nn.Module 的 buffer 默认建在 CPU，而 emb/model 在 CUDA。
    # bio_on=True 时 Nucleoskeleton.frame[idx].add_(task_vec) 会 CPU⊕CUDA 崩。
    # 只搬 buffer，不进 parameters/optimizer，不改任何数值（frame 全零）。
    for _m in (blood, receptor, skeleton, recall, temp_mod, enzyme, chloroplast):
        if isinstance(_m, torch.nn.Module):
            _m.to(device)

    get_batch, vocab = load_data(
        a.data, a.seq_len, a.bs, False, a.token_cache)
    n0, K = a.n, a.split_k
    model = build_bio(a, cfg, vocab, n0, device)
    cells = all_cells(model)
    # ---- V1.003 酶学：每 expert slot 一个区室酶池 ----
    # 默认 enabled=False ⇒ factors 全 1.0、零事件 ⇒ 301 点 loss/eff 零污染不破
    enz = Enzymology(n_slots=min(int(getattr(a, "compose_k", 16)), 16),
                     seed=a.seed,
                     enabled=bool(cfg.enable and cfg.enzyme),
                     state_every=500)
    for c in cells:
        if hasattr(c, "gate_comp"):
            c.enzymology = enz
            c.enzyme_factors = None
    # ---- V1.004：构造酶速率控制器（纯 Python，不进 model.parameters）----
    # off / amp 档【完全不实例化】：opt 分组与 lr 逐位不变（零污染硬前置）。
    # cat 系强制关闭旧 `ci·term` 前向，避免两个机制混在一起。
    rate = None
    if a.enz_rate_mode in ("cat", "cat_het", "const", "lrx"):
        rate = EnzymeRateController(
            n_cells=len(cells), mode=a.enz_rate_mode, seed=a.seed,
            beta_const=a.enz_beta_const, beta_ind=a.enz_beta_ind,
            tau_e=a.enz_tau_e, k_max=a.enz_k_max,
            gamma_km=a.enz_gamma_km, freeze_cell=a.enz_freeze_cell,
            cost_alpha=a.enz_cost_alpha,
            scope=a.enz_scope, warmup=a.enz_warmup)
        for c in cells:
            if hasattr(c, "gate_comp"):
                c.enzymology = None       # 关 V1.003 的 ci·term 前向
                c.enzyme_factors = None
    params0 = sum(p.numel() for p in model.parameters())

    # 两类忆点：info 取第一个忆点方向，task 取零向量（真实张量、真实二分）
    with torch.no_grad():
        blood.add("info", cells[0].organelle_memory.detach()[0].cpu(), 0)
        blood.add("task", torch.zeros(a.d), 0, task_id="task-init")

    for c in cells:
        cc, _ = c.connect()
        c._split_c = cc.detach().clone()
        c._split_n0 = n0
        c._split_child_w = None
    M0 = cells[0].organelle_memory.detach()
    g = SoftGroups(a.d, n0, G=a.groups, tau=0.10).to(device)
    g.init_from_memories(M0)
    gid = g.membership(M0).argmax(-1)
    peers_of = {int(gg): (gid == gg).nonzero().flatten()
                for gg in range(a.groups)}
    reg = OrganelleRegistry(a.d, n0, n_max=n0 + K)
    reg.group_id[:n0] = gid.cpu()[:n0]

    warm = max(1, a.max_steps // 10)
    def lr_at(s):
        w = min(s / warm, 1.0)
        return a.lr * (0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * w))))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=0.01, betas=(0.9, 0.95))
    # ★ 只有 cat/cat_het/const 才把 gate_comp 拆成独立 param_group；
    #   lrx 故意不分组（它要的就是「不分组的等效 lr」对照）。
    if rate is not None and a.enz_rate_mode != "lrx":
        rate.rebind_groups(opt, cells, a.lr)
    bio_trace_path = os.path.join(a.outdir, "%s_bio.jsonl" % tag)
    bio_update = BioUpdate.from_args(
        a, opt, model, cells, a.lr, trace_path=bio_trace_path)
    clip, refresh_clip = make_clip(model, a.clip_mode)

    model.train()
    t0 = time.time()
    hist, events = [], []
    busy, wired = [], False
    clip_acc = []
    all_ids = list(range(len(cells)))

    print("[bio] cell=%s  bio_on=%s  gain=%s  params=%s  Δparams=%+d" %
          (a.cell_type, cfg.enable, cfg.gate, format(params0, ","),
           param_overhead(len(cells), a.compose_mode, n0, a.d, a.h_cell)
           if a.cell_type == "residual" else 0), flush=True)
    print("[step29] enz_rate_mode=%s  cells=%d  (off/amp 档逐位不变)" %
          (a.enz_rate_mode, len(cells)), flush=True)

    for step in range(a.max_steps):
        lr = lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = lr
        # ---- V1.003：酶状态 → 本步各区室催化系数（默认全 1.0）----
        if cfg.enable and hasattr(cells[0], "enzymology"):
            fs = enz.factors()
            for c in cells:
                if hasattr(c, "gate_comp"):
                    c.enzyme_factors = fs
        x, y = get_batch("train")
        x, y = x.to(device), y.to(device)

        # ---- 观测：嵌入张量（detach，绝不回传、绝不注入 model）----
        with torch.no_grad():
            B, T = x.shape
            pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            emb = model.in_proj(model.embed(x) + model.pos(pos)).detach()
        _x2, robs = receptor(emb)              # 默认 gain=0 → 恒等
        _bl, bobs = blood.flow_step(0, emb, step)
        _rc, robs2 = recall.recall(emb, blood)
        _sk = skeleton.carry("task-init", torch.zeros(a.d, device=device))
        _ez = enzyme(emb)                      # 默认恒等
        _cp = chloroplast(emb)                 # 默认恒等

        # ---- 主动分裂（与历史逐位同路径）----
        if step == a.split_at:
            b0, b1 = float(atp.balance), float(atp.balance)
            if a.cell_type == "v10":
                busy = torch.topk(reg.fitness[:n0], K).indices.tolist()
                opt = do_split_v10(model, cells, a, n0, K, reg, peers_of,
                                   device, lr, step, busy)
            else:
                opt, busy = do_split_B(model, cells, a, n0, K, reg,
                                       peers_of, device, lr, step)
                if a.cell_type == "residual":
                    for c in cells:
                        c.expand(K, busy, device)      # 扩 E1/E2 buffer
            refresh_clip()
            # ★ do_split_B 重建了 optimizer ⇒ 必须重新绑定 per-cell gate 分组，
            #   否则分裂后 1500 步的 k 全部失效（v9 的「测量点」教训）。
            if rate is not None and a.enz_rate_mode != "lrx":
                rate.rebind_groups(opt, cells, lr)
            if bio_update is not None:
                # split 重建 optimizer；按 shared+cell 重新分组并按稳定组名续接 phi。
                bio_update.rebind(opt, model, cells, lr)
            if rate is not None:
                rate.on_divide(step, busy)   # duplication + 谱系，全自主
            events.append(dict(step=step, event="divide", K=K,
                               parents=busy, cell_type=a.cell_type))
            tel.on_divide(step, n0, K, busy, a.cell_type, b0, b1)
            if cfg.enable:
                enz.on_divide(step, busy)      # 基因复制 + 漂变，谱系写事件
            print("[split] step %d  %d→%d  params=%s  cell=%s" %
                  (step, n0, n0 + K,
                   format(sum(p.numel() for p in model.parameters()), ","),
                   a.cell_type), flush=True)

        # ---- 主动接线（复用历史秩-1 接线）----
        if step >= a.split_at:
            with torch.no_grad():
                w_old = (cells[0]._last_mix[:, :n0]
                         if cells[0]._last_mix is not None else None)
                xc_ = emb.detach().float().cpu().reshape(-1, a.d)  # 恢复历史二维口径
                if w_old is not None:
                    act_ = w_old.detach().float().cpu().mean(0)[:n0]
                    mi_, vi_ = xc_.mean(0), xc_.var(0).mean()
                    reg.in_centroid[:n0].mul_(0.99).add_(mi_ * 0.01)
                    reg.in_var[:n0].mul_(0.99).add_(vi_ * 0.01)
                    reg.fitness[:n0].mul_(0.99).add_(act_ * 0.01)
                    reg.hit_input[:n0].mul_(0.99).add_(mi_ * 0.01)
                mem_cpu = cells[0].organelle_memory.detach().float().cpu()
                for k in range(K):
                    j = n0 + k
                    wc = reg.wire_in(j, xc_, mem_cpu, eta=0.02, cap=0.05)
                    for c in cells:
                        c._split_child_w[k] = wc
                if not wired:
                    for k in range(K):
                        j = n0 + k
                        parent = busy[k] if k < len(busy) else 0
                        peers = peers_of[int(reg.group_id[parent])]
                        for c in cells:
                            cc = c._split_c
                            cc[j, peers] = 1.0 / len(peers)
                            cc[j] /= cc[j].sum()
                    cw = [round(v, 5) for v in cells[0]._split_child_w.tolist()]
                    events.append(dict(step=step, event="wire", child_w=cw))
                    tel.on_wire(step, cw)
                    wired = True

        # ---- 临时专家模组：正锥 Δw 真正接进 cell 前向（默认全 0 → 跳过）----
        if step % a.temp_module_every == 0 and cells[0]._last_mix is not None:
            w_for_dw = cells[0]._last_mix
            tel.open_temp_module(temp_mod, step, w_for_dw, all_ids,
                                 ["info-0001", "task-task-init"])
            if a.cell_type == "residual":
                for c in cells:
                    c.temp_delta_w = temp_mod.delta_w.detach().clone()
            if cfg.temp_module_ttl == 0:
                tel.dissolve_temp_module(temp_mod, step, 0.0)
                if a.cell_type == "residual":
                    for c in cells:
                        if hasattr(c, "temp_delta_w"):
                            c.temp_delta_w = torch.zeros_like(c.temp_delta_w)

        # ---- 前向 / 反向（与 train_compose 逐位同路径）----
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            logits, diag = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                               y.view(-1))
        hs = model.heal_sum()
        if hs is not None:
            loss = loss + hs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nm, ne = clip(model.parameters(), 1.0)
        clip_acc.append((nm, ne))
        # ---- V1.004：酶 = 速率常数，落在 AdamW 的 param_group.lr 上。
        # 必须在 clip 之后 / step 之前：grad 已全局缩放，median 归一化不受影响。
        # 乘 .grad 会被 Adam 的 m̂/√v̂ 洗掉，所以不乘 grad。
        if rate is not None:
            rate.observe_step(cells, opt, lr, step)
        bio_snapshot = None
        if bio_update is not None:
            bio_update.observe(float(loss.item()))
            bio_snapshot = bio_update.step(lr)
            bio_update.write_trace(step, bio_snapshot)
        opt.step()
        # ---- V1.003：底物 = 各专家 slot 本步真实前向统计量 ----
        # 叶绿体转底物 / 线粒体给 ATP / 血液两类忆点汇合 / 产物抑制负反馈
        if cfg.enable and hasattr(cells[0], "enzymology"):
            obs = getattr(cells[0], "_substrate_obs", None) or [0.0] * enz.n_slots
            ratio = float(atp.balance) / max(1.0, float(cfg.budget_init))
            enz.observe(step, obs, atp_ratio=ratio,
                        blood_info=float(len(blood.points("info"))),
                        blood_task=float(len(blood.points("task"))))
            enz.flush(tel, step)

        # ---- 履历（step0 起向量化在 CPU 累积，与历史一致）----
        if step < a.split_at:
            with torch.no_grad():
                lm = cells[0]._last_mix
                if lm is not None:
                    act = lm.detach().float().cpu().mean(0)[:n0]
                    em = model.in_proj(
                        model.embed(x) + model.pos(
                            torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)))
                    xc2 = em.detach().float().cpu().reshape(-1, a.d)
                    mi, vi = xc2.mean(0), xc2.var(0).mean()
                    reg.in_centroid[:n0].mul_(0.99).add_(mi * 0.01)
                    reg.in_var[:n0].mul_(0.99).add_(vi * 0.01)
                    reg.fitness[:n0].mul_(0.99).add_(act * 0.01)
                    reg.hit_input[:n0].mul_(0.99).add_(mi * 0.01)

        lv = float(loss.item())
        if step % 10 == 0 or step == a.max_steps - 1:
            eff = diag.get("eff_organs")
            vn = diag.get("wiring_var_norm")
            cw = diag.get("composed_width")
            hist.append(dict(step=step, loss=lv, lr=lr,
                             eff=float(eff) if eff is not None else None,
                             var_norm=float(vn) if vn is not None else None,
                             composed_width=float(cw) if cw is not None else None,
                             clip_main=float(nm)))
            # step28f 只读探针：diag = 最后一层 cell 返回的 {**_diag, **extra}
            # extra 里本来就有 gate_comp / comp_norm，只是从没写进 history
            try:
                _gc = float(diag.get("gate_comp", 0.0))
                _cn = float(diag.get("comp_norm", 0.0))
                _ci = []
                for _m in model.modules():
                    _f = getattr(_m, "enzyme_factors", None)
                    if _f is not None:
                        _ci.extend(float(_v) for _v in _f)
                _cm = sum(_ci) / len(_ci) if _ci else 1.0
                hist[-1].update(gate_comp=_gc, comp_norm=_cn, ci_mean=_cm)
                if rate is not None and rate.trace:
                    _t = rate.trace[-1]
                    hist[-1].update(
                        k_mean=_t["k_mean"], k_std=_t["k_std"],
                        S_spread=_t["S_spread"], e_mean=_t["e_mean"],
                        H_min=_t["H_min"], H_max=_t["H_max"],
                        corr_k_invH=_t["corr_k_invH"])
                if bio_snapshot is not None:
                    hist[-1].update(
                        phi_sum=bio_snapshot.sum_phi,
                        phi_scale_std=bio_snapshot.scale_std,
                        phi_budget_gap=bio_snapshot.budget_gap,
                        phi_G=len(bio_update.keys))
            except Exception:
                hist[-1].update(gate_comp=-9.0, comp_norm=-9.0, ci_mean=-9.0)
            if step % 500 == 0:
                print("  step %5d  loss=%.4f  eff=%s  clip_main=%.3f" %
                      (step, lv, "%.1f" % eff if eff else "-", nm), flush=True)

        # ---- 规格 11/13：每 500 步写一次完整观测事件 ----
        if step % 500 == 0:
            eff_v = float(diag.get("eff_organs") or 0.0)
            vn_v = float(diag.get("wiring_var_norm") or 0.0)
            wd = cells[0]._last_mix
            all_nonzero = bool(wd is not None and (wd > 0).all())
            fit_min = float(reg.fitness[:n0].min()) if hasattr(reg, "fitness") else 0.0
            tel.on_cell_call(step, all_ids, eff_v, vn_v, cfg.gate)
            tel.on_mem_kind(step, len(blood.points("info")),
                            len(blood.points("task")), "info", "info-0001")
            tel.on_mem_kind(step, len(blood.points("info")),
                            len(blood.points("task")), "task", "task-0001",
                            "task-init")
            _bev = atp.spend(0.0, "observe", step)
            tel.on_budget(step, float(_bev["spent"]),
                          float(_bev["balance"]), "observe")
            tel.on_transport(step, bobs["transport"], bobs["layer"], bobs["gain"])
            tel.on_health(step, eff_v, vn_v, all_nonzero, fit_min)
            # 自噬 / 主动死亡：只做扫描，触发才写事件（规格 A 可证伪）
            if cfg.enable:
                for cid in range(min(len(cells), 4)):
                    ev = autophagy.propose(step, cid, 1.0 / max(1, len(cells)),
                                           fit_min, False)
                    if ev:
                        tel.events.append(ev)
                    ev2 = apoptosis.propose(step, cid, 1.0 / max(1, len(cells)),
                                            step)
                    if ev2:
                        tel.events.append(ev2)
            # 规格 7：训练期复制被禁止，写一条「禁止」记录（不假装发生）
            tel.on_clone(step, 0, True)

    val = evaluate(model, get_batch, a.eval_batches, device, bf16)
    params1 = sum(p.numel() for p in model.parameters())
    eff_last = next((h["eff"] for h in reversed(hist) if h["eff"] is not None), None)
    vn_last = next((h["var_norm"] for h in reversed(hist)
                    if h["var_norm"] is not None), None)
    cw_last = next((h["composed_width"] for h in reversed(hist)
                    if h["composed_width"] is not None), None)
    cm_mean = sum(q[0] for q in clip_acc) / max(1, len(clip_acc))

    # 规格 13：任何未解散的临时模组都必须在收尾解散（dissolved_at 必填）
    if temp_mod.active:
        tel.dissolve_temp_module(temp_mod, a.max_steps - 1, 0.0)
    tel.flush_to(events)
    validation = validate_events(events, strict=False, observation_only=True)
    lineage = replay_lineage(events)

    print("\n" + "=" * 68)
    print("  [%s] val = %.4f  ppl = %.4f  eff=%s  var_norm=%s" %
          (tag, val, math.exp(val), eff_last, vn_last))
    print("  params %s → %s   composed_width=%s   用时 %.1fs" %
          (format(params0, ","), format(params1, ","), cw_last,
           time.time() - t0))
    print("  events=%d  kinds=%d  观测层校验=%s  谱系=%s" %
          (len(events), len(validation["kinds_present"]),
           "PASS" if validation["pass"] else "FAIL", lineage))
    print("=" * 68)

    rec = dict(
        tag=tag, mode=a.mode, config=vars(a), vocab_size=vocab,
        params_start=params0, params_end=params1,
        val_loss=val, ppl=math.exp(val),
        eff_organs_last=eff_last, var_norm_last=vn_last,
        composed_width=cw_last,
        clip_mode=a.clip_mode, clip_main_mean=cm_mean,
        compose_k=a.compose_k, h_cell=a.h_cell,
        compose_mode=a.compose_mode if a.cell_type == "residual" else None,
        avg_step_ms=1000.0 * (time.time() - t0) / max(1, a.max_steps),
        elapsed_sec=time.time() - t0,
        bio=cfg.as_dict(),
        bio_param_overhead=(param_overhead(len(cells), a.compose_mode,
                                           n0, a.d, a.h_cell)
                            if a.cell_type == "residual" else 0),
        event_schema=tel.schema(),
        telemetry_validation=validation,
        lineage=lineage,
        disabled_primitives=([] if cfg.enable else
                             ["autophagy", "apoptosis", "reset",
                              "原语前向注入(V1.002)"]),
        events=events, history=hist,
        enz_rate_mode=a.enz_rate_mode,
        enz_scope=(a.enz_scope if rate is not None else "off"),
        enz_rate_trace=(rate.trace if rate is not None else []),
        enz_rate_lineage=(rate.lineage if rate is not None else []),
        enz_rate_summary=(rate.summary() if rate is not None else None),
        bio_update_summary=(bio_update.summary() if bio_update is not None else None),
        bio_trace=(bio_trace_path if bio_update is not None else None),
        runtime=dict(device=str(device), bf16=bool(bf16),
                     cpu_threads=torch.get_num_threads(),
                     interop_threads=torch.get_num_interop_threads(),
                     token_cache=(a.token_cache or None)),
    )
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "%s.json" % tag)
    json.dump(rec, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[saved] %s" % out)

    if bio_update is not None:
        bio_update.close()
        if a.bio_require_active:
            try:
                bio_update.require_active(a.bio_min_scale_std)
                print("[G-PHI] PASS max_scale_std=%.6g" %
                      bio_update.summary()["max_scale_std"])
            except RuntimeError as exc:
                print("[G-PHI] MECHANISM_NOT_ACTIVE: %s" % exc, file=sys.stderr)
                raise SystemExit(7)

    # ---- 零污染自检（规格 12 红线 / 改名 B 的硬前置）----
    if a.zero_check:
        ref = os.path.join(a.outdir, "st25_s0_B.json")
        if not os.path.exists(ref):
            print("  [zero-check] 缺 %s，跳过（不假装通过）" % ref)
        else:
            import json as _json
            r = _json.load(open(ref))
            ha = {h["step"]: h for h in r["history"]}
            hb = {h["step"]: h for h in hist}
            cs = sorted(set(ha) & set(hb))
            mx = max(abs(ha[s]["loss"] - hb[s]["loss"]) for s in cs)
            dv = val - r["val_loss"]
            print("  [zero-check] 对齐点=%d  loss max|Δ|=%.3e  Δval=%+.3e" %
                  (len(cs), mx, dv))
            print("  [zero-check] %s" %
                  ("PASS → 观测层/生物学旁路零污染" if mx == 0.0 and dv == 0.0
                   else "FAIL → 有混淆变量，先查 RNG / 参数 / clip"))


if __name__ == "__main__":
    main()
