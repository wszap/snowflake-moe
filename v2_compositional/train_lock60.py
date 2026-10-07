# -*- coding: utf-8 -*-
"""
train_lock60.py —— 锁 6.0 真机训练 + 三变体消融

用法：
    python train_lock60.py --smoke                       # 合成数据端到端自检（先跑这个）
    python train_lock60.py --n 384 --delta-scale 0.5     # 正式
    python train_lock60.py --n 384 --delta-scale 1.0     # 第二档

⚠ 真机训练循环需要你接入真实数据（TinyStories）：
   把 lock47 的 train_one() 搬进来，或设置 --data 指向你的数据文件。
   本文件的 --smoke 模式不依赖外部数据，可立即验证全链路。
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snowflake_B import SnowflakeB                              # noqa: E402
from ablate_permute import eval_variants, ClusterTask          # noqa: E402
from run_metrics import RunMetrics                            # noqa: E402

def _pick_device():
    """自动设备选择：GPU 优先，无 GPU 则用 CPU。

    曾因【静默】回退 CPU，导致在按时长计费的服务器上白烧钱（见 L-018）。
    现在改成【显式】回退 —— 自动选，但启动即大声宣告用的是什么，
    并给出诊断与降配建议，不让人跑半天才发现。
    """
    import os
    cu = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[device] GPU OK  {name}  {gb:.1f}GB  "
              f"torch={torch.__version__} cuda={cu}", flush=True)
        return "cuda"
    print("[device] WARN 未检测到 CUDA -> 回退 CPU（速度约慢 50~100 倍）",
          flush=True)
    _hip = getattr(torch.version, "hip", None)
    if _hip:
        print(f"[device]      识别为 ROCm/HIP 分支（DCU），hip={_hip}", flush=True)
    elif cu is None:
        print("[device]      原因：装的是 CPU 版 torch"
              "（torch.version.cuda 为 None）", flush=True)
        print("[device]      修复：pip install torch --index-url "
              "https://download.pytorch.org/whl/cu121", flush=True)
    else:
        print(f"[device]      原因：torch 带 CUDA {cu}，"
              f"但驱动/运行时不可用", flush=True)
    print(f"[device]      CPU 核数 {os.cpu_count() or 1}；"
          f"建议 --scale tiny 并调小 --bs", flush=True)
    return "cpu"


DEVICE = _pick_device()

# --------------------------------------------------------------------------
# 规模预设（按 64GB 显存定档）
# --------------------------------------------------------------------------
# 之前所有实验都在 0.3M~1.8M 参数，你自己记录过：1.8M→0.5M 两种架构 PPL 都稳在
# 10.01~10.02 ⇒ 没吃满容量 ⇒ 架构差异淹没在噪声里。64GB 下应开大。
SCALE_PRESETS = {
    #              N    d    h    L  cells   r    P    bs   lr
    "tiny":   dict(n=384, d=128, h=32,  L=4,  cells=4, r=4,  P=32,  bs=64,  lr=3e-4),
    "mid":    dict(n=384, d=256, h=64,  L=8,  cells=4, r=4,  P=32,  bs=64,  lr=2e-4),
    "large":  dict(n=512, d=512, h=128, L=12, cells=4, r=4,  P=32,  bs=32,  lr=1e-4),
    "extreme":dict(n=1024,d=1024,h=256, L=16, cells=4, r=4,  P=32,  bs=16,  lr=8e-5),
}


# --------------------------------------------------------------------------
# 分层 LM（把 SnowflakeB 叠成语言模型）—— B 型低秩，见 D-004
# --------------------------------------------------------------------------
class SnowflakeLM(nn.Module):
    def __init__(self, vocab_size, d=128, n=384, L=4, h=32,
                 n_cells=4, seq_len=128, chunk_size=32, rank=4,
                 temp=5.0, hinge_mode="loss", use_hinge_low=False,
                 log_sharp=True, min_eff_organs=0, seed=0):
        super().__init__()
        self.d, self.L, self.n_cells = d, L, n_cells
        self.embed = nn.Embedding(vocab_size, d)
        self.in_proj = nn.Linear(d, d, bias=False)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                SnowflakeB(d, n_organelles=n, h=h, rank=rank,
                           seq_len=seq_len, chunk_size=chunk_size,
                           temp=temp, hinge_mode=hinge_mode,
                           use_hinge_low=use_hinge_low,
                           log_sharp=log_sharp,
                           min_eff_organs=min_eff_organs,
                           seed=seed * 1000 + li * 10 + ci)
                for ci in range(n_cells)
            ]) for li in range(L)
        ])
        self.head = nn.Linear(d, vocab_size)

    def forward(self, tokens, perm=None, force_mix=None):
        x = self.in_proj(self.embed(tokens))
        B, T, d = x.shape
        diag_last = {}
        for layer in self.layers:
            outs = []
            for cell in layer:
                y, info = cell(x.reshape(B * T, d), perm=perm, force_mix=force_mix)
                outs.append(y.reshape(B, T, d))
                diag_last = info
            x = sum(outs) / len(outs)
        return self.head(x), diag_last

    def heal_sum(self):
        s = None
        for layer in self.layers:
            for cell in layer:
                if cell.heal_term is not None:
                    s = cell.heal_term if s is None else s + cell.heal_term
        return s

    def agg_diag(self):
        # ★ 必须与 run_metrics.STEP_FIELDS 对齐，否则采集会漏字段。
        #   核心量 eff_organs / wiring_var_norm 见 L-023。
        keys = ["in_band", "cos_mean", "cos_std", "cos_max", "connect_rowsum",
                "wiring_ent", "wiring_variance", "wiring_var_norm",
                "eff_organs", "mix_sharp"]
        out = {k: [] for k in keys}
        for layer in self.layers:
            for cell in layer:
                st = cell.stats()
                for k in keys:
                    if k in st:
                        out[k].append(st[k])
        return {k: float(np.mean(v)) for k, v in out.items() if v}


# --------------------------------------------------------------------------
def smoke(args):
    """合成数据端到端自检：不依赖外部数据，验证全链路 + 判据能检出分工。"""
    print("=" * 78, flush=True)
    print("[smoke] 合成任务：K 簇 / 每簇一个真专家（已知有分工）", flush=True)
    print("=" * 78, flush=True)
    torch.manual_seed(0)
    K, D, H, SEP = 8, 64, 32, 2.0
    task = ClusterTask(n_cls=K, d=D, h=H, sep=SEP)
    # N=K=8 时 temp* ≈ 1.8（见 temp_scan.py）；4.0 会偏 one-hot
    model = SnowflakeB(D, n_organelles=K, h=H, rank=args.rank,
                       seq_len=1, chunk_size=1,
                       hinge_mode=args.hinge_mode,
                       temp=getattr(args, "temp", 1.8)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    for s in range(args.smoke_steps):
        tot = 0.0
        for xb, yb, _ in task.batches(bs=256):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            o, dg = model(xb)
            loss = loss_fn(o, yb)
            if model.heal_term is not None:
                loss = loss + model.heal_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        if s % 100 == 0 or s == args.smoke_steps - 1:
            st = model.stats()
            print(f"  step {s:>4} loss={tot:.5f} in_band={st.get('in_band',0):.3f} "
                  f"cos={st.get('cos_mean',0):.3f}±{st.get('cos_std',0):.3f} "
                  f"wiring_var={st.get('wiring_variance',0):.5f} "
                  f"sharp={st.get('mix_sharp',0):.3f}", flush=True)

    # 三变体
    @torch.no_grad()
    def run(mode, n_rep=1):
        L = []
        for _ in range(n_rep):
            for xb, yb, _ in task.batches(bs=512, shuffle=False):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                if mode == "learned":
                    o, _ = model(xb)
                elif mode == "constant":
                    o, _ = model(xb, force_mix=model.mix_ema)
                else:
                    C = xb.shape[0] // max(1, model.chunk_size)
                    o, _ = model(xb, perm=torch.randperm(C))
                L.append(loss_fn(o, yb).item())
        return float(np.mean(L))

    l_l = run("learned")
    l_c = run("constant")
    l_p = run("permuted", n_rep=args.n_perm)

    print(f"\n{'变体':<28}{'MSE':>14}{'mix_gain':>14}", flush=True)
    print("-" * 56, flush=True)
    print(f"  {'learned（学出配方）':<26}{l_l:>14.6f}{l_c - l_l:>14.6f}", flush=True)
    print(f"  {'permuted（破坏配对）':<26}{l_p:>14.6f}{l_c - l_p:>14.6f}", flush=True)
    print(f"  {'constant（完全退化）':<26}{l_c:>14.6f}{0.0:>14.6f}", flush=True)
    core = l_p - l_l
    print(f"\n  核心判据 loss(permuted) − loss(learned) = {core:+.6f}", flush=True)
    print("  判定：" + ("✅ 检出分工" if core > 1e-4
                        else "⚠ 未检出（先确认构造，别急着说'没有分工'）"), flush=True)
    return core


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--n-cells", type=int, default=4)
    ap.add_argument("--delta-scale", type=float, default=0.5)
    ap.add_argument("--chunk-size", type=int, default=128,
                    help="满秩 delta + N=384 时 P=128 才打平；P=32 是 3.25x 更贵")
    ap.add_argument("--hinge-mode", default="loss", choices=["loss", "mult", "none"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=300)
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--data", default=None, help="TinyStories 路径；None 则用合成数据")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--scale", default=None, choices=list(SCALE_PRESETS),
                    help="规模预设，覆盖 n/d/h/L/cells/chunk。64GB 推荐 mid 或 large")
    ap.add_argument("--no-bf16", dest="bf16", action="store_false",
                    default=True,
                    help="关闭 bf16。⚠ DCU/ROCm 上实测 bf16 比 fp32 慢 "
                         "2 倍（0.02ms vs 0.01ms），建议必关")
    ap.add_argument("--compile", action="store_true", help="torch.compile（首次慢，之后快）")
    ap.add_argument("--rank", type=int, default=4,
                    help="B 型低秩 rank。⚠ 不能开大：r=8→1.078x，r=32→1.312x（比 Top-K 更贵）")
    ap.add_argument("--temp", type=float, default=5.0,
                    help="softmax 温度。⚠ 必须随 N 缩放：N=8→1.8，N=384→5.09。"
                         "规格的 2.0 在 N=384 下会掉进均匀平均陷阱")
    ap.add_argument("--erase-frac", type=float, default=0.0,
                    help="擦除比例 0~1：随机把该比例的 organelle_memory 清零。"
                         "★ 注意：这是【破坏身份】，推理时【不会】保持性能，"
                         "必须重训练才能恢复（见 L-036）")
    ap.add_argument("--erase-ckpt", default="",
                    help="从该 ckpt 加载后执行擦除再重训练")
    ap.add_argument("--retrain-steps", type=int, default=0,
                    help="擦除后重训练步数（配合 --erase-frac 使用）")
    ap.add_argument("--fix-mojibake", action="store_true",
                    help="清洗 UTF-8 双重编码(mojibake)。"
                         "实测 vocab=100 应为 92，清洗后 loss 应更低"
                         "（见 L-031）")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--release-steps", type=int, default=0,
                    help="release test：训练后关掉 min_eff 再训 N 步，"
                         "判定涌现(<25%%) vs 强制(>60%%)。见 L-021"
                         "★ 这是判断『模型真想拼』的唯一实验")
    ap.add_argument("--save-ckpt", default="",
                    help="训练后保存 checkpoint 的路径")
    ap.add_argument("--eval-only", default="",
                    help="只跑评估，需配合 --save-ckpt 存过的 ckpt"
                         "ablation 崩了也能从 ckpt 恢复评估，不用重训")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="最大步数，0=跑完整个 epoch。"
                         "摸底建议 3000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logdir", default="output/runs",
                    help="证据落盘目录。每次运行自动写 record.json + "
                         "steps.csv，并追加 master.csv")
    ap.add_argument("--min-eff", type=int, default=0,
                    help="内生最低有效器官数 K（0=关闭，默认）。\n"
                         "实测 sharp=0.982 ⇒ 有效器官数 1.04（384选1）= 选专家。\n"
                         "设 8 ⇒ 每样本真正用到 8 个器官 = 拼专家。\n"
                         "单边：已够分散的样本不动。解析解、可微。见 D-005")
    ap.add_argument("--hinge-low", action="store_true",
                    help="开启下侧回复力（cos<0.3 推回）。cos 向下漂移时开")
    a = ap.parse_args()

    if a.scale:
        p = SCALE_PRESETS[a.scale]
        a.n, a.d, a.h, a.L, a.n_cells = p["n"], p["d"], p["h"], p["L"], p["cells"]
        a.chunk_size, a.lr = p["P"], p["lr"]
        if a.scale:
            a.bs = p["bs"]
        print(f"[scale={a.scale}] N={a.n} d={a.d} h={a.h} L={a.L} "
              f"cells={a.n_cells} P={a.chunk_size} lr={a.lr}", flush=True)

    if DEVICE == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB  "
              f"bf16={a.bf16}", flush=True)
    else:
        # CPU 上 bf16 支持差且常更慢；并自动降配，缩短等待
        if a.bf16:
            print("[device]      CPU 下自动关闭 bf16", flush=True)
            a.bf16 = False
        if a.scale is None:
            a.scale = "tiny"
            p = SCALE_PRESETS[a.scale]
            a.n, a.d, a.h, a.L, a.n_cells = p["n"], p["d"], p["h"], p["L"], p["cells"]
            a.chunk_size, a.lr = p["P"], p["lr"]
            print(f"[device]      CPU 下自动降为 --scale tiny "
                  f"(N={a.n} d={a.d} h={a.h})", flush=True)

    print(f"[cfg] N={a.n} d={a.d} h={a.h} L={a.L} cells={a.n_cells} "
          f"delta_scale={a.delta_scale} chunk={a.chunk_size} "
          f"hinge={a.hinge_mode} temp={a.temp} rank={a.rank} device={DEVICE}", flush=True)

    # FLOPs 账（跑前先看，别跑完才发现更贵）
    # ★ B 型低秩公式（不是满秩）。用满秩会算出 3.25x 而实际 0.664x，
    #   导致"比 Top-K 更贵"的错误结论（见 L-027）
    _r = a.rank
    per_tok = (a.n * a.d * _r) / a.chunk_size + a.d * _r + _r * a.h + a.d * a.h
    base = 4 * a.d * a.h
    print(f"[FLOPs] 拼专家/token={per_tok:,.0f}  Top-K k=4={base:,.0f}  "
          f"ratio={per_tok/base:.3f}x" +
          ("  ⚠ 比 Top-K 更贵" if per_tok > base else "  ✅ 更便宜"), flush=True)

    if a.smoke:
        smoke(a)
        return

    if a.data is None:
        print("\n[STOP] 正式训练需要 --data 指向 TinyStories。", flush=True)
        print("  先跑：python train_lock60.py --smoke", flush=True)
        print("  再跑：python train_lock60.py --scale mid --data <路径> "
              "--delta-scale 0.5", flush=True)
        print("\n  规模预设（64GB 显存）：", flush=True)
        for k, v in SCALE_PRESETS.items():
            print(f"    {k:<8} N={v['n']:<5} d={v['d']:<5} h={v['h']:<4} "
                  f"L={v['L']:<3} P={v['P']:<4} bs={v['bs']}", flush=True)
        return

    # ---- 正式训练（骨架；数据加载按你的 D 盘那套接入）----
    print(f"\n[train] 数据 {a.data}", flush=True)

    def load_data():
        """字符级加载（服务器 tinystories_100mb.txt 实测 vocab=92）。

        已用 probe_data.py 诊断：字符级、V=92、embedding 仅 23k 参数，
        不会喧宾夺主（占 tiny 主体 3.6M 的 0.65%）。
        若你更想沿用历史脚本的 tokenization，把这里换掉即可。
        """
        raw = open(a.data, "rb").read()
        txt = raw.decode("utf-8", errors="ignore")
        # ★ UTF-8 双重编码（mojibake）清洗：实测 vocab=100 而应为 92，
        #   多出的 8 个是 Â Ã â œ 这类噪声字符。它们把常见标点拆成多个
        #   token，污染 vocab 并抬高 loss/ppl。见 L-031
        if a.fix_mojibake:
            try:
                cand = txt.encode("latin-1").decode("utf-8")
                n0, n1 = len(set(txt)), len(set(cand))
                if n1 < n0:                 # 字符数变少 ⇒ 修复有效
                    print(f"[data] mojibake 修复: vocab {n0} -> {n1}",
                          flush=True)
                    txt = cand
                else:
                    print(f"[data] mojibake 修复无效({n0}->{n1})，保留原文",
                          flush=True)
            except (UnicodeEncodeDecodeError, UnicodeDecodeError):
                print("[data] 非双重编码，跳过修复", flush=True)
        chars = sorted(set(txt))
        stoi = {c: i for i, c in enumerate(chars)}
        # 报告非常规字符，便于人工确认
        _odd = [c for c in chars if ord(c) > 0x2100 or
                (0x80 <= ord(c) <= 0xFF)]
        if _odd:
            print(f"[data] 非常规字符 {len(_odd)} 个: "
                  f"{''.join(_odd[:20])!r}", flush=True)
        ids = torch.tensor([stoi[c] for c in txt], dtype=torch.long)
        V = len(chars)
        n = int(len(ids) * 0.9)
        tr, va = ids[:n], ids[n:]
        print(f"[data] {len(ids):,} tokens  vocab={V}  "
              f"train={len(tr):,} val={len(va):,}", flush=True)

        def get_batch(split):
            src = tr if split == "train" else va
            ix = torch.randint(len(src) - a.seq_len - 1, (a.bs,))
            x = torch.stack([src[i:i + a.seq_len] for i in ix])
            y = torch.stack([src[i + 1:i + 1 + a.seq_len] for i in ix])
            return x, y

        class _Iter:
            def __init__(self, split): self.split = split
            def __iter__(self): return self
            def __next__(self): return get_batch(self.split)

        return _Iter("train"), _Iter("val"), V

    train_iter, val_iter, vocab_size = load_data()

    # ---- 论文证据采集器（自动记 git/config/数据指纹/每步时序/消融）----
    rm = RunMetrics(vars(a), script=__file__, seed=getattr(a, "seed", 0),
                    tags=[a.scale or "custom", "train"], root=a.logdir)
    try:
        rm.attach_data(a.data, vocab_size=vocab_size,
                       split_method="sequential", split_seed=0,
                       doc_boundary_aware=False)
    except Exception as e:
        print(f"[WARN] 数据指纹采集失败（该 run 不能进主表）: {e}", flush=True)

    model = SnowflakeLM(vocab_size, d=a.d, n=a.n, L=a.L, h=a.h,
                        n_cells=a.n_cells, seq_len=a.seq_len,
                        chunk_size=a.chunk_size, rank=a.rank,
                        temp=a.temp, hinge_mode=a.hinge_mode,
                        use_hinge_low=a.hinge_low,
                        min_eff_organs=a.min_eff,
                        # ★ 必须传！否则 --seed 无效，
                        #   10 seeds 会全部用 seed=0 初始化 ⇒ 统计检验方差被
                        #   严重低估，CI 虚窄，结论不可信（见 L-027）
                        seed=a.seed).to(DEVICE)
    if a.bf16 and DEVICE == "cuda":
        model = model.to(torch.bfloat16)
    if a.compile:
        model = torch.compile(model)

    n_param = sum(p.numel() for p in model.parameters())
    print(f"[model] 参数量 {n_param:,}  显存占用约 "
          f"{n_param*4*4/1024**3:.2f}GB（含 Adam m,v + grad）", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    # ---- 擦除实验：破坏 organelle_memory（身份/连接），保留 W1_U/W2_U（能力）----
    # 见 L-036：新架构下忆点=连接本体，擦除≠旧架构的"删事实知识"。
    #   · 推理时擦除【不会】保持性能（清零后 logit=0 反而权重上升）
    #   · 但【可重训练恢复】—— 只丢了身份，能力还在
    #   · 全擦除(1.0) ⇒ 完全崩溃（wiring 均匀 ⇒ 退化为平均专家）
    if a.erase_ckpt:
        ck = torch.load(a.erase_ckpt, map_location=DEVICE)
        model.load_state_dict(ck["model"])
        print(f"[erase] 已从 {a.erase_ckpt} 恢复", flush=True)
    if a.erase_frac > 0:
        with torch.no_grad():
            g = torch.Generator().manual_seed(a.seed)
            for lyr in model.layers:
                for c in lyr:
                    n_org = c.organelle_memory.shape[0]
                    k = int(n_org * a.erase_frac)
                    if k <= 0:
                        continue
                    idx = torch.randperm(n_org, generator=g)[:k]
                    c.organelle_memory[idx] = 0.0
        print(f"[erase] 已清零 {a.erase_frac:.0%} 的 organelle_memory"
              f"（身份/连接）；W1_U/W2_U【保留】", flush=True)
        if a.erase_frac >= 1.0:
            print("[erase] ⚠ 全擦除 ⇒ wiring 将退化为均匀分布 ⇒ "
                  "模型等价于单一平均专家（预期完全崩溃）", flush=True)
    ce = nn.CrossEntropyLoss()
    print("[ready] 模型/优化器就绪，进入训练循环（若此后无输出，"
          "看 nvidia-smi：利用率>0 即正常运行）", flush=True)

    for ep in range(a.epochs):
        model.train()
        if a.max_steps and ep == 0:
            print(f"[cfg] 本次最多 {a.max_steps} 步", flush=True)
        t0 = time.time()
        for step, (xb, yb) in enumerate(train_iter):
            if a.max_steps and step >= a.max_steps:
                break
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if DEVICE == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=a.bf16):
                logits, _ = model(xb)
                # ★ get_batch 里 y 已经是 x 的 next-token（y[j]=x[j+1]），
                #   所以这里【不能再 shift】。再 shift 会错位 1 个 token，
                #   学成 x[j]→x[j+2]，loss 与 ppl 虚高（见 L-027）
                loss = ce(logits.reshape(-1, logits.size(-1)),
                          yb.reshape(-1))
                hs = model.heal_sum()
                if hs is not None:
                    loss = loss + hs
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 10 == 0 and step % 100 != 0:
                # 轻量心跳：不调 loss.item()（DCU 上每次同步数十 ms，见 L-020）
                print(f"  ...step {step} {(time.time()-t0):.0f}s", flush=True)
            if step % 100 == 0:
                dg = model.agg_diag()
                print(f"  ep{ep} step{step:>5} loss={loss.item():.4f} "
                      f"in_band={dg.get('in_band',0):.3f} "
                      f"wiring_var={dg.get('wiring_variance',0):.5f} "
                      f"cos={dg.get('cos_mean',0):.3f} "
                      f"sharp={dg.get('mix_sharp',0):.3f} "
                      f"eff={dg.get('eff_organs',0):.2f} "
                      f"{time.time()-t0:.0f}s", flush=True)
                # ← 论文证据自动采集（每 100 步）
                rm.log_step(step, model, loss=loss.item(), lr=a.lr,
                            tokens_seen=step * a.bs * a.seq_len, epoch=ep)
                t0 = time.time()

    # ---- release test：关掉约束继续训，判定"真想拼"还是"被摁着"----
    # ★ 判断拼专家成立与否的【唯一】实验（见 L-021 / D-005）
    if a.release_steps > 0:
        print("\n" + "=" * 76, flush=True)
        print("[release test] 关闭 min_eff 继续训练 —— 区分『真想拼』与『被摁着』",
              flush=True)
        print("=" * 76, flush=True)
        g0 = model.agg_diag()
        eff_on, sharp_on = g0.get("eff_organs", 0), g0.get("mix_sharp", 0)
        for lyr in model.layers:
            for c in lyr:
                c.min_eff_organs = 0        # 关约束
        for st_rel in range(a.release_steps):
            # ★ 必须在【训练】数据上继续训，用 val_iter 是数据泄漏
            xb, yb = next(train_iter)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=a.bf16):
                logits, _ = model(xb)
            loss = ce(logits.reshape(-1, logits.size(-1)),
                      yb.reshape(-1))
            for lyr in model.layers:
                for c in lyr:
                    if c.heal_term is not None:
                        loss = loss + c.heal_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if st_rel % 50 == 0:
                print(f"  ...release step {st_rel}", flush=True)
        g1 = model.agg_diag()
        eff_off, sharp_off = g1.get("eff_organs", 0), g1.get("mix_sharp", 0)
        onehot = 1.0 / a.n
        # ★ 主判据改用 eff（整体分布），sharp 仅作参考。
        #   N=384 下 sharp 只看最大值，对尾部噪声极敏感：
        #   实测 sharp 回弹 39.8%，但 eff 31.63→32.96 几乎不变（还略增），
        #   两者方向相反。eff 才真正代表"用了几个器官"。见 L-029
        rebound_eff = (eff_on - eff_off) / max(eff_on - 1.0, 1e-9)
        rebound_sharp = (sharp_off - sharp_on) / max(sharp_on - onehot, 1e-9)
        print(f"  约束开启: sharp={sharp_on:.4f}  eff={eff_on:.2f}", flush=True)
        print(f"  约束关闭: sharp={sharp_off:.4f}  eff={eff_off:.2f}", flush=True)
        print(f"  回弹(eff   主判据): {rebound_eff:+.1%}  "
              f"(0%=保持, 100%=弹回 one-hot)", flush=True)
        print(f"  回弹(sharp 仅参考): {rebound_sharp:+.1%}  "
              f"[N 大时对尾部噪声敏感]", flush=True)
        if abs(rebound_eff) < 0.25:
            v = "OK 真想拼：模型自发学会组合，论文可写『涌现』"
        elif abs(rebound_eff) < 0.60:
            v = "WARN 部分内化：需更长训练或降低 K 再测"
        else:
            v = "FAIL 被摞着：本质是选专家，论文只能写『架构强制』"
        print(f"  ★ 判定(依据 eff): {v}", flush=True)
        release = dict(release_sharp_on=sharp_on, release_sharp_off=sharp_off,
                       release_eff_on=eff_on, release_eff_off=eff_off,
                       release_rebound=float(rebound_eff),
                       release_rebound_sharp=float(rebound_sharp),
                       release_verdict=v)
        # release 之后【不】恢复约束：后续 ablation 要在无约束状态下评估，
        # 否则测的是"被摁着"的模型，判据失真。
        print("  [note] 约束保持关闭，后续 ablation 在无约束状态下评估", flush=True)
    else:
        release = {}

    # ---- 保存 checkpoint（让 ablation 崩了也能恢复评估）----
    if a.save_ckpt:
        try:
            torch.save({"model": model.state_dict(), "cfg": vars(a)},
                       a.save_ckpt)
            print(f"[ckpt] 已保存 -> {a.save_ckpt}", flush=True)
        except Exception as e:
            print(f"[WARN] ckpt 保存失败: {e}", flush=True)

    if a.eval_only:
        # 只评估模式：从 ckpt 恢复，跳过训练
        ck = torch.load(a.eval_only, map_location=DEVICE)
        model.load_state_dict(ck["model"])
        print(f"[eval-only] 已从 {a.eval_only} 恢复", flush=True)

    # ---- 训练后：三变体消融（唯一能证伪"分工"的实验）----
    # ★ try/except 兜底：ablation 崩也要保证训练数据落盘。
    #   曾因 DEV 未定义崩溃，且 finalize 只在最后 => 3000 步全丢（L-026）。
    print("\n[ablation] 三变体评估（learned / constant / permuted）", flush=True)
    try:
        abl = rm.ablation(model, val_iter, ce, n_batch=50, n_perm=a.n_perm,
                          device=DEVICE, seed=a.seed)
        print(f"  learned={abl['loss_learned']:.5f}  "
              f"constant={abl['loss_constant']:.5f}  "
              f"permuted={abl['loss_permuted']:.5f}", flush=True)
        print(f"  核心判据 = {abl['core_stat']:+.6f}  "
              f"CI=[{abl['core_stat_ci_lo']:+.4f}, "
              f"{abl['core_stat_ci_hi']:+.4f}]  "
              f"d_z={abl['cohens_dz']:.3f}", flush=True)
    except Exception as e:
        import traceback
        print(f"[WARN] ablation 失败（训练数据仍会落盘）: {e}", flush=True)
        traceback.print_exc()
        abl = {}

    final = dict(abl)
    final.update(release)
    final["flops_per_token_theory"] = int(per_tok)
    final["flops_ratio"] = model.layers[0][0].flops_ratio_vs_topk()
    rid = rm.finalize(model, final=final,
                      verdicts=RunMetrics.auto_verdicts(final))
    print(f"\n[证据已落盘] output/runs/{rid}/", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
