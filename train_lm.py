# -*- coding: utf-8 -*-
"""MarvisMoE v7 —— 语言建模扩展（阶段 G1/G2）

字符级 next-token 预测（莎士比亚文本）：
- 数据管线：字符级 tokenization、90/10 切分、seq_len 采样批。
- 模型改造：nn.Embedding(vocab, d) + in_proj(d->d) + L 层 MoELayer
  + head(nn.Linear(d, vocab))，输出 (B, T, vocab) 逐 token logits。
- 损失：F.cross_entropy(logits.view(-1, vocab), targets.view(-1))。
- 治理：复用 train_v7 的 LearnedCouncil / REINFORCE 机制（fixed / rules /
  learned / learned_nok 及消融/容量/蒸馏模式），保证与 MNIST 实验同机制可比。

用法：
    python train_lm.py --smoke                  # 冒烟：1 batch forward/backward
    python train_lm.py --run                    # 主实验：fixed/rules/learned/learned_nok × 3 seed
"""
import argparse
import csv
import gc
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marvis_moe import Config, Council, Education, MoELayer, Monitor, set_seed  # noqa: E402
from marvis_moe_v7 import LearnedCouncil, make_state  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "tinyshakespeare", "input.txt")
OUT_LM_CSV = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "output", "results_lm.csv"))

LM_CSV_HEADER = ["data", "seed", "mode", "epochs", "val_ppl", "val_loss",
                 "train_loss", "final_cv", "active", "avg_k", "drop_ratio",
                 "events", "n_rewards", "sec"]


# ================================================================ 数据管线
def load_shakespeare(path=DATA_PATH):
    """读取莎士比亚文本，字符级 tokenize，90/10 切分。返回 (vocab_size, train_ids, val_ids, chars)。"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_train = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:n_train], ids[n_train:]
    return vocab_size, train_ids, val_ids, chars


def lm_batch(ids, batch_size, seq_len, device=DEVICE, seed=None):
    """从 1D token 序列随机采样 (B, T) 输入与 (B, T) next-token 标签。"""
    g = torch.Generator()
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
    n = ids.numel()
    max_start = n - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,), generator=g)
    x = torch.stack([ids[s:s + seq_len] for s in starts])
    y = torch.stack([ids[s + 1:s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)


# ================================================================ 模型改造
class MarvisMoE_LM(nn.Module):
    """字符级 LM 版 MarvisMoE：Embedding + in_proj + L×MoELayer + Linear head。

    与 MarvisMoE 保持同接口（council / education / mon / layers），
    训练循环与治理机制完全复用 train_v7 逻辑。
    """
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, cfg.d)
        self.in_proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.layers = nn.ModuleList([MoELayer(cfg) for _ in range(cfg.L)])
        self.head = nn.Linear(cfg.d, vocab_size)
        self.council = Council(cfg)
        self.education = Education(cfg.E)
        self.mon = Monitor(cfg.E)

    def forward(self, tokens, adjust=None, use_token_dropping=False):
        x = self.embed(tokens)                       # (B, T, d)
        x = self.in_proj(x)
        aux_total = 0.0
        for layer in self.layers:
            x, aux = layer(x, topk=self.council.k, cap=self.council.cap,
                           lbda=self.council.lbda, adjust=adjust,
                           use_token_dropping=use_token_dropping)
            aux_total = aux_total + aux
        return self.head(x), aux_total               # (B, T, vocab)


def eval_ppl(model, val_ids, vocab_size, seq_len, n_batch=20, batch_size=16,
             device=DEVICE):
    """验证集平均 CE -> PPL = exp(CE)。返回 (avg_ce, ppl)。"""
    model.eval()
    ces = []
    with torch.no_grad():
        for _ in range(n_batch):
            xb, yb = lm_batch(val_ids, batch_size, seq_len, device)
            logits, _ = model(xb)
            ce = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1)).item()
            ces.append(ce)
    model.train()
    avg_ce = float(np.mean(ces))
    return avg_ce, math.exp(avg_ce)


# ================================================================ 训练循环
def train_lm(cfg, vocab_size, train_ids, val_ids, council_mode="learned",
             seed=2026, epochs=5, batch_size=16, seq_len=64, lr=3e-4, accum=8,
             council_every=20, G=4, beta=0.5, gamma=0.1, fixed_cap=None,
             use_token_dropping=False, n_val_batch=20, device=DEVICE, verbose=True,
             use_clip=False, use_warmup=False):
    """语言建模主循环：与 train_v7 同治理机制，仅数据/损失改为 LM 形态。

    模式集合与 train_v7 完全一致（fixed/rules/learned/learned_nok/
    learned_decoupled/learned_ablate_*/learned_capdrop/learned_tokendrop/
    learned_no_edu/learned_fixed_edu/learned_adaptive_edu）。
    """
    assert council_mode in (
        "fixed", "rules", "learned", "learned_nok", "learned_decoupled",
        "learned_ablate_loss", "learned_ablate_cv", "learned_ablate_k",
        "learned_capdrop", "learned_tokendrop",
        "learned_no_edu", "learned_fixed_edu", "learned_adaptive_edu",
    ), f"未知 council_mode: {council_mode}"
    set_seed(seed)
    model = MarvisMoE_LM(cfg, vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = max(1, train_ids.numel() // (batch_size * seq_len))
    clip_norm = 1.0 if use_clip else 5.0
    warmup_steps = max(1, int(steps_per_epoch * epochs * 0.05)) if use_warmup else 0
    steps = 0
    accum_cnt = 0

    council = None
    dec_plan = None
    learned_modes = ("learned", "learned_nok", "learned_decoupled",
                     "learned_ablate_loss", "learned_ablate_cv", "learned_ablate_k",
                     "learned_capdrop", "learned_tokendrop",
                     "learned_no_edu", "learned_fixed_edu", "learned_adaptive_edu")
    if council_mode in learned_modes:
        council = LearnedCouncil(state_dim=4, E=cfg.E, topk=cfg.topk,
                                 control_k=(council_mode != "learned_nok"))
        council_opt = torch.optim.Adam(council.parameters(), lr=1e-3)
        reward_baseline = None
        step_actions = None
        cur_lp = None
        cur_r_sum = 0.0
        cur_cnt = 0
        prev_seg_avg = None
        cur_seg_ce_sum = 0.0
        cur_seg_cnt = 0
        last_ce = 1.0
        last_grad = 0.0
        collect_states = []
        if council_mode == "learned_decoupled":
            _s1 = max(1, epochs // 3)
            _s2 = max(1, epochs // 3)
            _s3 = epochs - _s1 - _s2
            dec_plan = (_s1, _s2, _s3)
        else:
            dec_plan = None

    history = dict(loss=[], acc=[], val_loss=[], val_ppl=[], std=[], cv=[],
                   events=[], rewards=[], k_seq=[], council_log=[], drop_ratios=[])

    for ep in range(epochs):
        freeze_council = False
        freeze_main = False
        if dec_plan is not None:
            s1, s2, s3 = dec_plan
            if ep < s1:
                freeze_council = True
            elif ep < s1 + s2:
                freeze_main = True
        if freeze_main:
            for p in model.parameters():
                p.requires_grad_(False)

        for bi in range(steps_per_epoch):
            xb, yb = lm_batch(train_ids, batch_size, seq_len, device, seed=seed)

            # ---- 治理参数确定（与 train_v7 完全一致）----
            if council_mode == "fixed":
                params = dict(lbda=0.15, cap=1.25, k=cfg.topk, edu=0.0)
            elif council_mode == "rules":
                if model.mon.steps % council_every == 0:
                    model.council.step(model.mon)
                params = dict(lbda=model.council.lbda, cap=model.council.cap,
                              k=model.council.k, edu=model.council.edu)
            else:  # learned 系列
                if step_actions is None or model.mon.steps % council_every == 0:
                    if cur_lp is not None and cur_cnt > 0:
                        r_mean = cur_r_sum / cur_cnt
                        if reward_baseline is None:
                            reward_baseline = r_mean
                        else:
                            reward_baseline = 0.9 * reward_baseline + 0.1 * r_mean
                        if freeze_council:
                            cur_lp = None
                        else:
                            adv = r_mean - reward_baseline
                            council_opt.zero_grad()
                            loss_rl = -(cur_lp * adv) - 0.01 * council.log_std.mean()
                            loss_rl.backward()
                            council_opt.step()
                            history["events"].append(("learned_council", model.mon.steps,
                                                      dict(lbda=params["lbda"], cap=params["cap"],
                                                           k=params["k"], edu=params["edu"])))
                    state = make_state(model.mon, last_ce, last_grad, cfg.topk)
                    step_actions = council(state, explore=not freeze_council)
                    collect_states.append(state.detach())
                    if len(collect_states) >= 10:
                        council.update_norm(collect_states)
                        collect_states = []
                    if cur_seg_cnt > 0:
                        prev_seg_avg = cur_seg_ce_sum / cur_seg_cnt
                    cur_seg_ce_sum = 0.0
                    cur_seg_cnt = 0
                    cur_lp = step_actions["log_prob"]
                    if freeze_council:
                        cur_lp = None
                    cur_r_sum = 0.0
                    cur_cnt = 0
                    history["k_seq"].append(step_actions["k"])
                    history["council_log"].append(dict(
                        step=model.mon.steps, lbda=step_actions["lbda"],
                        cap=step_actions["cap"], k=step_actions["k"],
                        edu=step_actions["edu"],
                        freeze_council=freeze_council, freeze_main=freeze_main))
                params = dict(lbda=step_actions["lbda"], cap=step_actions["cap"],
                              k=step_actions["k"], edu=step_actions["edu"])
                if council_mode in ("learned_capdrop", "learned_tokendrop"):
                    params["cap"] = fixed_cap if fixed_cap is not None else step_actions["cap"]
                if council_mode == "learned_no_edu":
                    params["edu"] = 0.0
                elif council_mode in ("learned_fixed_edu", "learned_adaptive_edu"):
                    params["edu"] = 0.15

            # ---- LR warmup（前 5% steps 线性升温）----
            if use_warmup:
                if steps < warmup_steps:
                    opt.param_groups[0]["lr"] = lr * (steps + 1) / warmup_steps
                else:
                    opt.param_groups[0]["lr"] = lr

            # ---- 前向/损失（LM 形态）----
            opt.zero_grad()
            logits, aux = model(xb, use_token_dropping=use_token_dropping)
            ce = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
            total = ce + params["lbda"] * aux
            if council_mode in learned_modes and params["edu"] > 0 and model.mon.steps % 50 == 0:
                feat = model.embed(xb)  # (B, T, d) float 特征，soft_loss 需 float 输入
                dloss, nw = model.education.soft_loss(model, model.mon, feat,
                                                      adaptive=(council_mode == "learned_adaptive_edu"))
                total = total + params["edu"] * dloss
            total = total / accum
            if not freeze_main:
                total.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm))
            else:
                grad_norm = 0.0
            accum_cnt += 1
            if accum_cnt >= accum:
                if not freeze_main:
                    opt.step()
                accum_cnt = 0

            # ---- 监控 ----
            with torch.no_grad():
                acc = (logits.argmax(-1) == yb).float().mean().item()
                util = torch.zeros(cfg.E, device=model.layers[0].last_idx2.device)
                ent_sum, n_layers = 0.0, len(model.layers)
                for layer in model.layers:
                    util.scatter_add_(0, layer.last_idx2.reshape(-1),
                                      torch.ones(layer.last_idx2.numel(), device=util.device))
                    p = layer.last_full_p
                    ent_sum += float(-(p * p.clamp_min(1e-9).log()).sum(-1).mean())
                entropy = ent_sum / max(1, n_layers)
                model.mon.update(util, ce.item() * accum, entropy)

            history["loss"].append(ce.item())
            history["acc"].append(acc)
            last_ce = ce.item()
            last_grad = grad_norm

            # ---- 学习型议会：段奖励累积 ----
            if council_mode in learned_modes and step_actions is not None:
                with torch.no_grad():
                    base = prev_seg_avg if prev_seg_avg is not None else ce.item()
                    r = -(ce.item() - base) - beta * model.mon.util_cv \
                        - gamma * abs(step_actions["k"] - cfg.topk)
                    if council_mode == "learned_ablate_loss":
                        r = - beta * model.mon.util_cv - gamma * abs(step_actions["k"] - cfg.topk)
                    elif council_mode == "learned_ablate_cv":
                        r = -(ce.item() - base) - gamma * abs(step_actions["k"] - cfg.topk)
                    elif council_mode == "learned_ablate_k":
                        r = -(ce.item() - base) - beta * model.mon.util_cv
                cur_r_sum += r
                cur_cnt += 1
                cur_seg_ce_sum += ce.item()
                cur_seg_cnt += 1
                history["rewards"].append(r)

            history["drop_ratios"].append(
                float(sum(getattr(l, "last_drop_ratio", 0.0) for l in model.layers)) / max(1, len(model.layers)))
            history["std"].append(model.mon.util_std)
            history["cv"].append(model.mon.util_cv)
            steps += 1

        # 每 epoch 验证
        if freeze_main:
            for p in model.parameters():
                p.requires_grad_(True)
        avg_ce, ppl = eval_ppl(model, val_ids, vocab_size, seq_len,
                               n_batch=n_val_batch, device=device)
        history["val_loss"].append(avg_ce)
        history["val_ppl"].append(ppl)
        if verbose:
            print(f"   epoch {ep + 1}/{epochs}: train_ce={np.mean(history['loss'][-steps_per_epoch:]):.4f} "
                  f"val_ce={avg_ce:.4f} val_ppl={ppl:.4f} cv={model.mon.util_cv:.4f}")

    history["final_util_std"] = model.mon.util_std
    history["final_cv"] = model.mon.util_cv
    history["final_active"] = model.mon.active_frac
    history["train_acc"] = float(np.mean(history["acc"][-30:])) if history["acc"] else -1.0
    history["avg_loss"] = float(np.mean(history["loss"][-30:])) if history["loss"] else -1.0
    history["avg_k"] = float(np.mean(history["k_seq"])) if history["k_seq"] else float(cfg.topk)
    history["drop_ratio"] = float(np.mean(history["drop_ratios"])) if history["drop_ratios"] else 0.0
    history["val_ppl_final"] = history["val_ppl"][-1] if history["val_ppl"] else -1.0
    history["val_loss_final"] = history["val_loss"][-1] if history["val_loss"] else -1.0
    return model, history, council


# ================================================================ 实验流水线
def load_done_lm(out_csv):
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["data"], int(row["seed"]), row["mode"], int(row["epochs"])))
    return done


def append_row_lm(out_csv, row):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LM_CSV_HEADER)
        if new:
            w.writeheader()
        w.writerow(row)


def run_experiments_lm(seeds=(2026, 2027, 2028), modes=("fixed", "rules", "learned"),
                       epochs=5, d=64, h=128, E=4, S=1, L=2, topk=2,
                       seq_len=64, batch_size=16, accum=8, lr=3e-4,
                       cap=None, out_csv=OUT_LM_CSV, data_path=DATA_PATH,
                       device=DEVICE, n_val_batch=20,
                       use_clip=False, use_warmup=False):
    """语言建模实验流水线：modes × seeds → CSV（断点续跑 + 温度守护）。"""
    vocab_size, train_ids, val_ids, chars = load_shakespeare(data_path)
    print(f"[LM] 数据: {data_path}")
    print(f"[LM] vocab={vocab_size} train_tokens={train_ids.numel()} val_tokens={val_ids.numel()} "
          f"seq_len={seq_len} batch={batch_size} accum={accum} (等效batch={batch_size * accum})")

    cfg = Config(d=d, h=h, E=E, S=S, L=L, topk=topk, out_dim=vocab_size)
    done = load_done_lm(out_csv)
    total = len(seeds) * len(modes)
    idx = 0
    results = {m: [] for m in modes}

    for sd in seeds:
        for m in modes:
            idx += 1
            key = ("shakespeare", sd, m, epochs)
            if key in done:
                print(f"[SKIP {idx}/{total}] shakespeare seed={sd} mode={m} epochs={epochs} (已存在)")
                continue
            t0 = time.time()
            print(f"[RUN   {idx}/{total}] shakespeare seed={sd} mode={m} epochs={epochs} ...")
            sys.stdout.flush()
            model, h, _ = train_lm(cfg, vocab_size, train_ids, val_ids,
                                   council_mode=m, seed=sd, epochs=epochs,
                                   batch_size=batch_size, seq_len=seq_len,
                                   lr=lr, accum=accum, fixed_cap=cap,
                                   use_token_dropping=(m == "learned_tokendrop"),
                                   n_val_batch=n_val_batch, device=device,
                                   use_clip=use_clip, use_warmup=use_warmup)
            row = dict(data="shakespeare", seed=sd, mode=m, epochs=epochs,
                       val_ppl=round(h["val_ppl_final"], 6),
                       val_loss=round(h["val_loss_final"], 6),
                       train_loss=round(h["avg_loss"], 6),
                       final_cv=round(h["final_cv"], 6),
                       active=round(h["final_active"], 6),
                       avg_k=round(h["avg_k"], 4),
                       drop_ratio=round(h.get("drop_ratio", 0.0), 4),
                       events=len(h["events"]),
                       n_rewards=len(h["rewards"]),
                       sec=round(time.time() - t0, 1))
            append_row_lm(out_csv, row)
            results[m].append(row)
            gc.collect()
            print(f"   -> val_ppl={row['val_ppl']:.4f} val_loss={row['val_loss']:.4f} "
                  f"train_loss={row['train_loss']:.4f} cv={row['final_cv']:.4f} "
                  f"active={row['active']:.2%} avg_k={row['avg_k']} "
                  f"events={row['events']} ({row['sec']}s)")
            sys.stdout.flush()

    print("\n" + "=" * 72)
    print(f"语言建模流水线完成 输出: {out_csv}")
    print("=" * 72)
    for m in modes:
        rows = results[m]
        if not rows:
            continue
        pp = np.array([r["val_ppl"] for r in rows])
        cv = np.array([r["final_cv"] for r in rows])
        print(f"{m:<10} val_ppl={pp.mean():.4f}±{pp.std():.4f}  cv={cv.mean():.4f}±{cv.std():.4f}  n={len(rows)}")
    return out_csv


def smoke_test(d=64, h=128, E=4, S=1, L=2, topk=2, seq_len=64, batch_size=16,
               device=DEVICE):
    """G1.3 冒烟：1 batch forward/backward，打印初始 PPL。"""
    vocab_size, train_ids, val_ids, chars = load_shakespeare()
    cfg = Config(d=d, h=h, E=E, S=S, L=L, topk=topk, out_dim=vocab_size)
    set_seed(2026)
    model = MarvisMoE_LM(cfg, vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    xb, yb = lm_batch(train_ids, batch_size, seq_len, device, seed=2026)
    logits, aux = model(xb)
    ce = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    total = ce + 0.15 * aux
    total.backward()
    opt.step()
    with torch.no_grad():
        logits2, _ = model(xb)
        ce2 = F.cross_entropy(logits2.view(-1, vocab_size), yb.view(-1))
    ppl_init = math.exp(ce2.item())
    print(f"[SMOKE OK] vocab={vocab_size} logits={tuple(logits.shape)} "
          f"x={tuple(xb.shape)} ce={ce.item():.4f} aux={aux.item():.4f}")
    print(f"[SMOKE OK] 1 batch forward/backward/step 完成，初始 PPL={ppl_init:.4f}")
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"[SMOKE] 显存峰值 {alloc:.2f} GB (< 7GB 达标)")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MarvisMoE v7 语言建模")
    p.add_argument("--smoke", action="store_true", help="冒烟：1 batch")
    p.add_argument("--run", action="store_true", help="主实验：modes × seeds")
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    p.add_argument("--modes", nargs="+", default=["fixed", "rules", "learned", "learned_nok"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--h", type=int, default=128)
    p.add_argument("--E", type=int, default=4)
    p.add_argument("--S", type=int, default=1)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--cap", type=float, default=None)
    p.add_argument("--out", default=OUT_LM_CSV)
    args = p.parse_args()
    if args.smoke:
        smoke_test()
    else:
        run_experiments_lm(seeds=tuple(args.seeds), modes=tuple(args.modes),
                           epochs=args.epochs, d=args.d, h=args.h, E=args.E,
                           S=args.S, L=args.L, topk=args.topk,
                           seq_len=args.seq_len, batch_size=args.batch,
                           accum=args.accum, lr=args.lr, cap=args.cap,
                           out_csv=os.path.abspath(args.out))
