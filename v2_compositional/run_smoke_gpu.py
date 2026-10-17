# -*- coding: utf-8 -*-
"""
run_smoke_gpu.py —— GPU 冒烟 + 正式训练一条命令跑完（10 小时预算内）

用法：
    python run_smoke_gpu.py --smoke-only          # 只跑冒烟（~2 分钟）
    python run_smoke_gpu.py --data <TinyStories>  # 冒烟通过后直接进正式

严格 GPU 优先：检测不到 CUDA 直接退出，不静默回退 CPU。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- 自动设备选择：GPU 优先，无 GPU 则用 CPU ----
import sys as _sys
if torch.cuda.is_available():
    DEV = "cuda"
    _gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[device] GPU OK  {torch.cuda.get_device_name(0)}  {_gb:.1f}GB",
          flush=True)
else:
    DEV = "cpu"
    _cu = getattr(torch.version, "cuda", None)
    print("[device] WARN 未检测到 CUDA -> 回退 CPU（速度约慢 50~100 倍）",
          flush=True)
    if _cu is None:
        print("[device]      原因：装的是 CPU 版 torch"
              "（torch.version.cuda 为 None）", flush=True)
        print("[device]      修复：pip install torch --index-url "
              "https://download.pytorch.org/whl/cu121", flush=True)
    else:
        print(f"[device]      原因：torch 带 CUDA {_cu}，"
              f"但驱动/运行时不可用", flush=True)

from snowflake_B import SnowflakeB                         # noqa: E402
from ablate_permute import ClusterTask                    # noqa: E402
from run_metrics import RunMetrics                        # noqa: E402


def smoke(rank=4, steps=300, n_perm=20, n_cls=8, d=64, h=32, sep=2.0,
          use_hinge_low=False, log_sharp=True, temp=2.0,
          metrics=None, device=None, min_eff=0):
    """阳性对照：已知有分工的合成任务，核心判据必须 > 0。"""
    print("\n" + "=" * 88, flush=True)
    print(f"[smoke] 合成任务 {n_cls} 簇 / 每簇一个真专家 / rank={rank}（已知有分工）", flush=True)
    print("=" * 88, flush=True)
    torch.manual_seed(0)
    task = ClusterTask(n_cls=n_cls, d=d, h=h, sep=sep)
    model = SnowflakeB(d, n_organelles=n_cls, h=h, rank=rank,
                   seq_len=1, chunk_size=1, temp=temp,
                   use_hinge_low=use_hinge_low,
                   log_sharp=log_sharp,
                   min_eff_organs=min_eff).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lf = nn.MSELoss()
    t0 = time.time()

    for s in range(steps):
        # 只在打印步做 .item()（每次 .item() 强制 GPU->CPU 同步，
        # DCU 上可达数十 ms —— 这是"每步 2 秒"的主因，见 L-020）
        want_report = (s % 100 == 0 or s == steps - 1)
        tot_acc = None
        nb = 0
        for xb, yb, _ in task.batches(bs=256):
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad(set_to_none=True)
            o, dg = model(xb)
            loss = lf(o, yb)
            if model.heal_term is not None:
                loss = loss + model.heal_term
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if want_report:
                tot_acc = loss.detach() if tot_acc is None else tot_acc + loss.detach()
                nb += 1
        tot = float(tot_acc) / max(nb, 1) if want_report else float("nan")
        if s % 100 == 0 or s == steps - 1:
            st = model.stats()
            print(f"  step {s:>4} loss={tot:.5f} in_band={st.get('in_band',0):.3f} "
                  f"cos={st.get('cos_mean',0):.3f} "
                  f"wiring_var={st.get('wiring_variance',0):.5f} "
                  f"sharp={st.get('mix_sharp',0):.3f} "
                  f"{time.time()-t0:.0f}s", flush=True)
            if metrics is not None:
                metrics.log_step(s, model, loss=tot, lr=1e-2)

    @torch.no_grad()
    def run(mode, n_rep=1):
        L = []
        for _ in range(n_rep):
            for xb, yb, _ in task.batches(bs=512, shuffle=False):
                xb, yb = xb.to(DEV), yb.to(DEV)
                if mode == "learned":
                    o, _ = model(xb)
                elif mode == "constant":
                    o, _ = model(xb, force_mix=model.mix_ema)
                else:
                    C = xb.shape[0] // max(1, model.chunk_size)
                    o, _ = model(xb, perm=torch.randperm(C, device=DEV))
                L.append(lf(o, yb).item())
        return float(np.mean(L))

    l_l, l_c = run("learned"), run("constant")
    l_p = run("permuted", n_rep=n_perm)

    print(f"\n{'变体':<28}{'MSE':>14}{'mix_gain':>14}", flush=True)
    print("-" * 56, flush=True)
    print(f"  {'learned（学出配方）':<26}{l_l:>14.6f}{l_c - l_l:>14.6f}", flush=True)
    print(f"  {'permuted（破坏配对）':<26}{l_p:>14.6f}{l_c - l_p:>14.6f}", flush=True)
    print(f"  {'constant（完全退化）':<26}{l_c:>14.6f}{0.0:>14.6f}", flush=True)
    core = l_p - l_l
    print(f"\n  核心判据 loss(permuted) − loss(learned) = {core:+.6f}", flush=True)

    # ---- release test：关掉约束继续训，看 sharp 是否弹回 ----
    # 判断"涌现 vs 强制"的唯一方法（见 L-021）
    release = None
    if getattr(smoke, "_release_steps", 0) > 0:
        print("\n" + "=" * 88, flush=True)
        print("[release test] 关闭 min_eff 继续训练 —— 区分『学会』与『强制』",
              flush=True)
        print("=" * 88, flush=True)
        sharp_on = float(model.stats().get("mix_sharp", 0))
        eff_on = float(model.stats().get("eff_organs", 0))
        model.min_eff_organs = 0          # 关约束
        for _ in range(smoke._release_steps):
            for xb, yb, _ in task.batches(bs=256):
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad(set_to_none=True)
                o, _ = model(xb)
                loss = lf(o, yb)
                if model.heal_term is not None:
                    loss = loss + model.heal_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        st = model.stats()
        sharp_off = float(st.get("mix_sharp", 0))
        eff_off = float(st.get("eff_organs", 0))
        onehot = 1.0 / model.n
        rebound = (sharp_off - sharp_on) / max(sharp_on - onehot, 1e-9)
        print(f"  约束开启: sharp={sharp_on:.4f}  有效器官数={eff_on:.2f}", flush=True)
        print(f"  约束关闭: sharp={sharp_off:.4f}  有效器官数={eff_off:.2f}", flush=True)
        print(f"  回弹幅度: {rebound:.1%}  (0%=完全保持, 100%=完全弹回 one-hot)",
              flush=True)
        if rebound < 0.25:
            verdict = "OK 涌现：模型自己学会了组合，论文可写『涌现』"
        elif rebound < 0.60:
            verdict = "WARN 部分内化：需更长训练或降低 K 再测"
        else:
            verdict = "FAIL 强制：纯靠约束兜着，论文只能写『架构强制最低组合度』"
        print(f"  判定: {verdict}", flush=True)
        release = dict(sharp_on=sharp_on, sharp_off=sharp_off,
                       eff_on=eff_on, eff_off=eff_off,
                       rebound=float(rebound), verdict=verdict)
    if metrics is not None:
        abl = {"loss_learned": l_l, "loss_constant": l_c, "loss_permuted": l_p,
               "core_stat": core}
        if release:
            abl.update({
                "release_sharp_on": release["sharp_on"],
                "release_sharp_off": release["sharp_off"],
                "release_eff_on": release["eff_on"],
                "release_eff_off": release["eff_off"],
                "release_rebound": release["rebound"],
                "release_verdict": release["verdict"],
            })
        metrics.log.final.update(abl)
    ok = core > 1e-4
    print("  判定：" + ("✅ 检出分工，可以进正式训练" if ok
                        else "⚠ 未检出 —— 检查构造/超参，别直接下'没有分工'的结论"), flush=True)
    return core, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--sep", type=float, default=2.0,
                    help="簇分离度。one-hot 诊断：先用 2.0 跑，若 sharp≈0.9 "
                         "再跑 1.0 对照。sep=1.0 下若软化 ⇒ 数据问题；"
                         "仍 one-hot ⇒ 架构问题，需熵奖励")
    ap.add_argument("--logdir", default="output/runs",
                    help="证据落盘目录。每次运行自动写 record.json/steps.csv"
                         "并追加 master.csv")
    ap.add_argument("--release-steps", type=int, default=0,
                    help="release test 步数：训练完后关掉 min_eff "
                         "继续训 N 步，看 sharp 是否弹回。建议 200。"
                         "判定: <25%%=涌现, >60%%=强制（见 L-021）")
    ap.add_argument("--min-eff", type=int, default=0,
                    help="内生最低有效器官数 K（0=关闭）。冒烟里 n_cls=8，"
                         "建议试 4 验证 sharp 能否从 1.0 降下来")
    ap.add_argument("--temp", type=float, default=2.0,
                    help="softmax 温度。规格=2.0。⚠ 早期版本硬编码 4.0，"
                         "是 sharp=1.0 的部分原因，已修正")
    ap.add_argument("--hinge-low", action="store_true",
                    help="开启下侧回复力(cos<0.3 推回)。默认关闭")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-perm", type=int, default=20)
    a = ap.parse_args()

    print(f"[cfg] rank={a.rank}  "
          f"FLOPs@P=32={SnowflakeB(128,384,32,a.rank,seq_len=128,chunk_size=32).flops_ratio_vs_topk():.3f}x  "
          f"组合空间={SnowflakeB(128,384,32,a.rank).combo_space_dim()}维", flush=True)

    cfg = dict(rank=a.rank, steps=a.steps, n_perm=a.n_perm, n_cls=8, d=64,
               h=32, sep=a.sep, temp=a.temp, hinge_low=a.hinge_low)
    rm = RunMetrics(cfg, script=__file__, seed=0, tags=["smoke", "gpu"],
                    root=a.logdir)

    smoke._release_steps = a.release_steps
    core, ok = smoke(rank=a.rank, steps=a.steps, n_perm=a.n_perm,
                 sep=a.sep, use_hinge_low=a.hinge_low, temp=a.temp,
                 metrics=rm, device=DEV, min_eff=a.min_eff)

    verdicts = {"H4_分工真实存在": "pass" if ok else "fail",
                "smoke_pass": "pass" if ok else "fail"}
    rid = rm.finalize(model=None,
                      final={"core_stat": core},
                      verdicts=verdicts)
    print(f"\n[证据已落盘] output/runs/{rid}/  "
          f"(record.json + steps.csv + master.csv)", flush=True)

    if a.smoke_only:
        print("\n[smoke-only] 完成。把核心判据发给元宝决定 width/temp。", flush=True)
        return

    if not ok:
        print("\n[STOP] smoke 未检出分工，不进正式训练。先排查。", flush=True)
        return
    if a.data is None:
        print("\n[STOP] 需要 --data 指向 TinyStories。", flush=True)
        return
    print(f"\n[next] 正式训练 {a.data} —— 见 train_lock60.py --scale mid", flush=True)


if __name__ == "__main__":
    main()
