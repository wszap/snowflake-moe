# -*- coding: utf-8 -*-
"""
paperplaneMoe / step25 —— 真·拼专家主训练入口

在仓库最新 main (0934ab31) 的 train_split.py / snowflake_v4.py 基础上改，
但【零改动现有源码】：用模块级替换把 make_cell() 里的 SnowflakeB 换成
compose_v10.ComposableExpertV10，其余数据/评估/主动分裂全复用原实现。

新增三件事（都来自仓库里已被实测证过的结论）：
  1. 换核心 cell：旧 B 型「先合参数 → 单个共享非线性」(秩 ≤ rank)
     → v10「各自非线性 → 加权求和」(等效宽度 k×h_cell 线性增长)
  2. 分组裁剪 --clip-mode grouped：把 layers.* 专家参数与主干分离裁剪，
     消除 step23 三 seed 证实的「新增参数经全局 clip_grad_norm_ 拖累主干」
  3. CRN 配对：同 seed 同 batch 序列，落盘额外诊断供 step25 判据脚本做
     配对 t / Holm / CUPED，奔着「显著」而不是单 seed 好看数

用法（在 v2_compositional/ 下）：
  python3 train_compose.py --data data_25mb.txt --cell-type v10 \
      --h-cell 4 --compose-k 16 --clip-mode grouped --seed 0 --tag st25_s0_V10iso
"""
import argparse
import functools
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_v3 import load_data, evaluate
from organelle_registry import OrganelleRegistry
from organelle_groups import SoftGroups
from train_split import do_split as do_split_B   # 基线档严格复用历史代码
from compose_v10 import ComposableExpertV10

import snowflake_v4
from snowflake_v4 import SnowflakeLMV4

_ORIG_B = snowflake_v4.SnowflakeB


# ---------------------------------------------------------------- 模型构建
def _make_v10_cls(a):
    """★ 用 wrapper 类覆盖 kw，而不是 functools.partial。

    snowflake_v4.make_cell() 的 kw 里已经带 chunk_size=1；若再用 partial
    传 chunk_size，调用时会 TypeError: multiple values for keyword argument。
    这里在 __init__ 里对同一 kw 字典做 update，覆盖而非重复传参。
    """
    class V10Cell(ComposableExpertV10):
        def __init__(self, **kw):
            kw.update(chunk_size=a.compose_chunk,   # 默认 1 = 与 B 档逐位同口径
                      h_cell=a.h_cell,
                      k=a.compose_k,
                      renorm=(a.compose_renorm == 1),
                      mem_budget_mb=a.mem_budget)
            super().__init__(wiring_mod=a.wiring_mod, **kw)
    return V10Cell


def install_cells(a):
    """make_cell() 引用的是模块全局名 SnowflakeB，替换即换核心 cell。"""
    if a.cell_type == "v10":
        snowflake_v4.SnowflakeB = _make_v10_cls(a)
    else:
        snowflake_v4.SnowflakeB = _ORIG_B


def build(a, vocab, n_start, device):
    install_cells(a)
    return SnowflakeLMV4(
        vocab_size=vocab, d=a.d, n=n_start, L=a.L, h=a.h,
        n_cells=a.n_cells, seq_len=a.seq_len, rank=a.rank, temp=a.temp,
        n_head=a.n_head, scaffold_top_k=a.scaffold_top_k,
        select_mode=a.select_mode, variant=a.variant,
        shared_mult=a.shared_mult, seed=a.seed,
        clone_temp=a.clone_temp, groups=0).to(device)


def all_cells(model):
    return [c for layer in model.layers for c in layer]


# ---------------------------------------------------------------- 分组裁剪
def make_clip(model, mode):
    """global = 历史原样；grouped = layers.* 与主干各自独立裁到 max_norm。

    grouped 的语义：主干的 clip_coef 不再被「专家库有多少参数」影响。
    """
    _orig = torch.nn.utils.clip_grad_norm_

    if mode == "global":
        def clip(params, max_norm=1.0, *args, **kw):
            n = _orig(params, max_norm, *args, **kw)
            return float(n), None
        return clip, lambda: None

    expert_ids = set()

    def refresh():
        expert_ids.clear()
        for name, p in model.named_parameters():
            if name.startswith("layers."):
                expert_ids.add(id(p))

    def clip(params, max_norm=1.0, *args, **kw):
        ps = list(params)                       # generator 只能消费一次
        main = [p for p in ps if id(p) not in expert_ids]
        exp = [p for p in ps if id(p) in expert_ids]
        nm = _orig(main, max_norm, *args, **kw) if main else 0.0
        ne = _orig(exp, max_norm, *args, **kw) if exp else 0.0
        return float(nm), float(ne)

    refresh()
    return clip, refresh


# ---------------------------------------------------------------- 分裂
def do_split_v10(model, cells, a, n0, K, reg, peers_of, device, lr, step, busy):
    """v10 只需扩 E1/E2（零初始化），其余接线/履历与历史完全一致。"""
    for k, par in enumerate(busy):
        j = n0 + k
        reg.in_centroid[j] = reg.in_centroid[par].clone()
        reg.hit_input[j] = reg.hit_input[par].clone()
        reg.in_var[j] = 0.0
        reg.fitness[j] = 0.0
        reg.group_id[j] = int(reg.group_id[par])
        reg.birth_step[j] = step
        reg.parent_id[j] = par
        reg.w_child[j] = 0.0017
        reg.active[j] = True
    reg.n = n0 + K
    for c in cells:
        c.expand(K, busy, device, child_w=0.0017)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=0.01, betas=(0.9, 0.95))
    return opt


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
    ap.add_argument("--mode", choices=["split"], default="split")
    ap.add_argument("--split-k", type=int, default=8)
    ap.add_argument("--split-at", type=int, default=1500)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--join-mode", choices=["share"], default="share")
    # ---- step25 新增 ----
    ap.add_argument("--cell-type", choices=["B", "v10"], default="v10")
    ap.add_argument("--h-cell", type=int, default=3)
    ap.add_argument("--compose-k", type=int, default=16)
    ap.add_argument("--compose-renorm", type=int, default=1)
    ap.add_argument("--compose-chunk", type=int, default=1,
                    help="v10 的 wiring 粒度；1=与 B 档完全同口径(慢)，32=可行性档(快)")
    ap.add_argument("--mem-budget", type=float, default=256.0)
    ap.add_argument("--wiring-mod", choices=["none", "norm", "center"],
                    default="none")
    ap.add_argument("--clip-mode", choices=["global", "grouped"],
                    default="grouped")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default="output/v4")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda"
    max_steps = a.max_steps
    tag = a.tag or ("st25_%s_s%d" % (a.cell_type, a.seed))

    # 数据（RESULTS.md 坑：必须显式传 --data）
    get_batch, vocab = load_data(a.data, a.seq_len, a.bs, False)
    n0, K = a.n, a.split_k
    n_start = n0
    model = build(a, vocab, n_start, device)
    cells = all_cells(model)
    params0 = sum(p.numel() for p in model.parameters())

    # 冻结 c（与 train_split.py 对称）；新细胞不进 softmax 抢名额
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

    warm = max(1, max_steps // 10)

    def lr_at(s):
        w = min(s / warm, 1.0)
        return a.lr * (0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * w))))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=0.01, betas=(0.9, 0.95))
    clip, refresh_clip = make_clip(model, a.clip_mode)

    model.train()
    t0 = time.time()
    step_t0 = time.time()
    start_loss = final_loss = None
    hist, events = [], []
    busy = []
    wired = False
    clip_acc = []

    for step in range(max_steps):
        lr = lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = get_batch("train")
        x, y = x.to(device), y.to(device)

        # ---- 主动分裂：由自身 fitness 选最忙的 K 个母体 ----
        if step == a.split_at:
            busy = torch.topk(reg.fitness[:n0], K).indices.tolist()
            if a.cell_type == "v10":
                opt = do_split_v10(model, cells, a, n0, K, reg, peers_of,
                                   device, lr, step, busy)
            else:
                opt, busy = do_split_B(model, cells, a, n0, K, reg,
                                       peers_of, device, lr, step)
            refresh_clip()                     # 参数对象重建 → 重算分组 id
            events.append(dict(step=step, event="divide", K=K,
                               parents=busy, cell_type=a.cell_type))
            print("[split] step %d  %d→%d  params=%s  cell=%s clip=%s" %
                  (step, n0, n0 + K,
                   format(sum(p.numel() for p in model.parameters()), ","),
                   a.cell_type, a.clip_mode), flush=True)

        # ---- 主动接线：份额由自身忆点命中率；秩-1 只落本群（一次）----
        if step >= a.split_at:
            with torch.no_grad():
                B, T = x.shape
                pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
                emb = model.in_proj(model.embed(x) + model.pos(pos)).reshape(-1, a.d)
                w_old = (cells[0]._last_mix[:, :n0]
                         if cells[0]._last_mix is not None else None)
                xc_ = emb.detach().float().cpu()
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
                        gid_p = int(reg.group_id[parent])
                        peers = peers_of[gid_p]
                        for c in cells:
                            cc = c._split_c
                            cc[j, peers] = 1.0 / len(peers)
                            cc[j] /= cc[j].sum()
                    events.append(dict(step=step, event="wire",
                                       child_w=[round(v, 5) for v in
                                                cells[0]._split_child_w.tolist()]))
                    wired = True

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
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
        opt.step()

        # ---- 履历从 step0 向量化累积在 CPU（分裂时 topk 才有意义）----
        if step < a.split_at:
            with torch.no_grad():
                lm = cells[0]._last_mix
                if lm is not None:
                    act = lm.detach().float().cpu().mean(0)[:n0]
                    B, T = x.shape
                    pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
                    em = model.in_proj(model.embed(x) + model.pos(pos))
                    xc2 = em.detach().float().cpu().reshape(-1, a.d)
                    mi, vi = xc2.mean(0), xc2.var(0).mean()
                    reg.in_centroid[:n0].mul_(0.99).add_(mi * 0.01)
                    reg.in_var[:n0].mul_(0.99).add_(vi * 0.01)
                    reg.fitness[:n0].mul_(0.99).add_(act * 0.01)
                    reg.hit_input[:n0].mul_(0.99).add_(mi * 0.01)

        lv = loss.item()
        if start_loss is None:
            start_loss = lv
        final_loss = lv
        if step % 10 == 0 or step == max_steps - 1:
            eff = diag.get("eff_organs")
            vn = diag.get("wiring_var_norm")
            cw = diag.get("composed_width")
            rec_step = dict(
                step=step, loss=lv, lr=lr,
                eff=float(eff) if eff is not None else None,
                var_norm=float(vn) if vn is not None else None,
                composed_width=float(cw) if cw is not None else None,
                clip_main=float(nm),
                clip_expert=(float(ne) if ne is not None else None))
            hist.append(rec_step)
            if step % 500 == 0:
                print("  step %5d  loss=%.4f  eff=%s  clip_main=%.3f" %
                      (step, lv, "%.1f" % eff if eff else "-", nm), flush=True)

    val = evaluate(model, get_batch, 20, device, bf16)
    params1 = sum(p.numel() for p in model.parameters())
    vn_last = next((h["var_norm"] for h in reversed(hist)
                    if h["var_norm"] is not None), None)
    eff_last = next((h["eff"] for h in reversed(hist)
                     if h["eff"] is not None), None)
    cw_last = next((h["composed_width"] for h in reversed(hist)
                    if h["composed_width"] is not None), None)
    cm_mean = sum(q[0] for q in clip_acc) / max(1, len(clip_acc))
    ce_vals = [q[1] for q in clip_acc if q[1] is not None]
    ce_mean = sum(ce_vals) / len(ce_vals) if ce_vals else None

    print("\n" + "=" * 68)
    print("  [%s] val = %.4f  ppl = %.4f   eff=%s  var_norm=%s" %
          (tag, val, math.exp(val), eff_last, vn_last))
    print("  params %s → %s   composed_width=%s   用时 %.1fs" %
          (format(params0, ","), format(params1, ","), cw_last,
           time.time() - t0))
    print("  clip: mode=%s  main_mean=%.3f  expert_mean=%s" %
          (a.clip_mode, cm_mean,
           "%.3f" % ce_mean if ce_mean is not None else "-"))
    step_ms = 1000.0 * (time.time() - step_t0) / max(1, max_steps)
    print("  chunk=%d  k=%d  h_cell=%d  avg_step=%.1f ms" %
          (a.compose_chunk, a.compose_k, a.h_cell, step_ms))
    print("=" * 68)

    os.makedirs(a.outdir, exist_ok=True)
    rec = dict(tag=tag, mode=a.mode, config=vars(a), vocab_size=vocab,
               params_start=params0, params_end=params1,
               start_loss=start_loss, final_loss=final_loss,
               val_loss=val, ppl=math.exp(val),
               eff_organs_last=eff_last, var_norm_last=vn_last,
               composed_width=cw_last,
               clip_mode=a.clip_mode, clip_main_mean=cm_mean,
               compose_chunk=a.compose_chunk, compose_k=a.compose_k,
               h_cell=a.h_cell, avg_step_ms=1000.0*(time.time()-step_t0)/max(1,max_steps),
               clip_expert_mean=ce_mean,
               elapsed_sec=time.time() - t0, events=events, history=hist)
    out = os.path.join(a.outdir, "%s.json" % tag)
    json.dump(rec, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[saved] %s" % out)


if __name__ == "__main__":
    main()
