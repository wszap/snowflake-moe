# -*- coding: utf-8 -*-
"""
run_metrics.py —— 论文级数据自动采集（接入训练/冒烟，一条命令全量落盘）

解决的问题：
    之前指标只打印到 stdout，跑完就没了。论文需要的
    git commit / config hash / 数据 sha256 / 环境指纹 / 每步时序 / 消融判据
    全部靠手工整理 —— 不可复现、不可审计。

    本模块把采集【写进运行文件】，每次运行自动产出：
        output/runs/<run_id>/record.json   完整记录（meta+data+steps+final+verdicts）
        output/runs/<run_id>/steps.csv     每步时序
        output/runs/master.csv             所有 run 的横向汇总

=============================================================================
用法（三行接入）
=============================================================================
    from run_metrics import RunMetrics
    rm = RunMetrics(cfg, script=__file__, seed=seed, root="output/runs")
    rm.attach_data("TinyStories.txt", tokenizer=..., vocab_size=V)

    for step, (xb, yb) in enumerate(train_iter):
        ...
        rm.log_step(step, model, loss=loss.item(), lr=..., tokens_seen=...)

    rm.finalize(model, eval_fn=..., verdicts={...})

=============================================================================
自动采集什么
=============================================================================
【meta】run_id、时间(CST/UTC)、脚本名、seed、tags、
        config 全量 + config_hash、python/torch/cuda 版本、GPU 型号、
        git commit + git_dirty
【data】数据 sha256、字节数、mtime、tokenizer、vocab_size、切分方式
【step】loss、lr、in_band、cos_mean、cos_std、wiring_ent、wiring_variance、
        mix_sharp、heal_term、显存峰值、吞吐(tok/s)、墙钟
【final】val/test ppl、参数量、理论+实测 FLOPs、延迟、显存峰值、
        **三变体消融**：loss_learned / loss_constant / loss_permuted、
        mix_gain、核心判据 + 95% CI、Cohen's d_z
【verdicts】H1~H7 预注册判据的 pass/fail

=============================================================================
字段注册表（防口径漂移）
=============================================================================
未注册的字段名会直接抛错。这条防的是 `wiring_ent` → `wiring_entropy`
这类改名导致的历史数据口径丢失。新增指标必须先在这里登记。
"""
import math
import os
import time

import numpy as np
import torch

from experiment_logger import (RunLogger, STEP_FIELDS, FINAL_FIELDS,  # noqa: E402
                               seeds_aggregate, paired_compare)

# ---- B 型（lock6.2）新增字段注册 ----
# ★ 必须与 snowflake_B.stats() 和 SnowflakeLM.agg_diag() 的 key 完全对齐。
#   漏注册会让 log_step 抛错（实测踩过：eff_organs / wiring_var_norm 缺失）。
#   新增指标【必须】在这里登记，否则整条证据链断。
STEP_FIELDS.update({
    "cos_mean": "float", "cos_std": "float", "cos_max": "float",
    "connect_rowsum": "float",
    "wiring_ent": "float",
    "wiring_variance": "float",
    # ★ 跨 N 可比的核心量（见 L-023）
    "wiring_var_norm": "float",
    "eff_organs": "float",
    "mix_sharp": "float",
    "in_band": "float",
    "heal_term": "float",
    "combo_dim": "float", "flops_ratio": "float",
    "throughput_tok_s": "float",
    "wall_sec": "float", "epoch": "int", "tokens_seen": "int",
    "lr": "float",
})
FINAL_FIELDS.update({
    "loss_learned": "float", "loss_constant": "float", "loss_permuted": "float",
    "loss_permuted_sd": "float",
    "core_stat": "float", "core_stat_ci_lo": "float", "core_stat_ci_hi": "float",
    "cohens_dz": "float", "p_core": "float", "n_permute_repeats": "int",
    "release_sharp_on": "float", "release_sharp_off": "float",
    "release_eff_on": "float", "release_eff_off": "float",
    "release_rebound": "float", "release_verdict": "str",
    "eff_organs_final": "float", "wiring_var_norm": "float",
    "mix_sharp_final": "float", "in_band_final": "float", "cos_mean_final": "float",
    "combo_dim": "float", "flops_ratio": "float",
    "sharp_warn_triggered": "int",
})


def gpu_stats():
    """GPU 实测：显存峰值 / 利用率 / 温度。失败返回 None（不阻断训练）。"""
    if not torch.cuda.is_available():
        return {}
    try:
        return {
            "gpu_mem_peak_mb": float(torch.cuda.max_memory_allocated() / 1024 ** 2),
        }
    except Exception:
        return {}


class RunMetrics:
    """包装 RunLogger，自动从 model 抽取架构诊断量。"""

    def __init__(self, cfg, script, seed=0, tags=None, root="output/runs",
                 sharp_warn=0.7):
        self.cfg = dict(cfg)
        self.cfg["seed"] = seed
        self.log = RunLogger(self.cfg, script=script, tags=tags, root=root)
        self.sharp_warn = sharp_warn
        self.t0 = time.time()
        self.tokens_seen = 0
        self.sharp_triggered = 0
        self._hist = []           # 每步 (loss_learned_proxy) 留作配对检验

    # ---------------------------------------------------------------- 数据
    def attach_data(self, path, **kw):
        return self.log.attach_data(path, **kw)

    # ---------------------------------------------------------------- 步
    def log_step(self, step, model=None, loss=None, lr=None,
                 tokens_seen=None, epoch=0, extra=None):
        rec = {"epoch": int(epoch),
               "wall_sec": time.time() - self.t0}
        if loss is not None:
            rec["train_loss"] = float(loss)
        if lr is not None:
            rec["lr"] = float(lr)
        if tokens_seen is not None:
            self.tokens_seen = int(tokens_seen)
        rec["tokens_seen"] = self.tokens_seen

        # 从 model 自动抽架构诊断
        if model is not None:
            st = self._model_stats(model)
            rec.update(st)
            if st.get("mix_sharp", 0) > self.sharp_warn:
                self.sharp_triggered += 1

        rec.update(gpu_stats())
        if extra:
            rec.update(extra)
        try:
            return self.log.log_step(step, **rec)
        except Exception as e:
            # 落盘失败不能拖垮训练（字段未注册会抛错，见 L-024）
            print(f"[WARN] log_step 失败（训练继续）: {e}", flush=True)
            print(f"       未注册字段: {sorted(rec.keys())}", flush=True)
            return None

    def _model_stats(self, model):
        """从 SnowflakeLM / SnowflakeB 抽诊断量。兼容两者。"""
        out = {}
        try:
            agg = model.agg_diag()          # SnowflakeLM
        except Exception:
            try:
                agg = model.stats()         # SnowflakeB
            except Exception:
                return out
        key_map = {
            "in_band": "in_band", "cos_mean": "cos_mean", "cos_std": "cos_std",
            "cos_max": "cos_max", "connect_rowsum": "connect_rowsum",
            "wiring_ent": "wiring_ent", "wiring_variance": "wiring_variance",
            "mix_sharp": "mix_sharp",
        }
        for src, dst in key_map.items():
            if src in agg:
                out[dst] = float(agg[src])
        # heal_term / 效率
        try:
            hs = model.heal_sum()
            if hs is not None:
                out["heal_term"] = float(hs.detach())
        except Exception:
            pass
        # combo_dim / flops_ratio 定义在【cell】上，不在 LM 上。
        # 优先取 layers[0][0]，回退 model 自身（单 cell 场景）。
        cell = None
        for layer in getattr(model, "layers", []):
            for c in layer:
                cell = c
                break
            if cell is not None:
                break
        for attr, key in (("combo_space_dim", "combo_dim"),
                          ("flops_ratio_vs_topk", "flops_ratio")):
            for obj in (cell, model):
                if obj is not None and hasattr(obj, attr):
                    try:
                        out[key] = float(getattr(obj, attr)())
                        break
                    except Exception:
                        pass
        # 吞吐
        if self.tokens_seen > 0:
            out["throughput_tok_s"] = self.tokens_seen / max(
                time.time() - self.t0, 1e-6)
        return out

    # ---------------------------------------------------------------- 三变体
    @torch.no_grad()
    def ablation(self, model, data_iter, loss_fn, n_batch=50, n_perm=20,
                 device=None, seed=0):
        """三变体消融：learned / constant / permuted。返回判据字典。

        这是唯一能证伪「分工真实存在」的实验，必须每次训练后跑。
        """
        dev = device or next(model.parameters()).device
        model.eval()
        # ★ generator 必须在【目标 device】上创建。
        #   DCU/HIP 伪装 cuda 时，CPU generator + cuda randperm 会报
        #   "Expected a 'cuda' device type for generator but found 'cpu'"
        #   见 L-030
        try:
            g = torch.Generator(device=dev)
        except (TypeError, RuntimeError, AttributeError):
            g = torch.Generator()          # 老版本 torch 不支持 device 参数
        g.manual_seed(seed)
        L = {"learned": [], "constant": [], "permuted": []}
        per_sample = {"learned": [], "permuted": []}

        # ★ loss 必须 reshape：CrossEntropyLoss 的 input 是 [N, C]，
        #   [bs, seq, V] 直接喂会被当成 (N=bs, C=seq) 而报错/算错（L-027）
        def _ce(o, y):
            return loss_fn(o.reshape(-1, o.size(-1)), y.reshape(-1))

        for bi, batch in enumerate(data_iter):
            if bi >= n_batch:
                break
            xb, yb = self._unpack(batch, dev)
            o, _ = model(xb)
            L["learned"].append(_ce(o, yb).item())

            o, _ = model(xb, force_mix=self._mix_ema(model))
            L["constant"].append(_ce(o, yb).item())

            # ★ C 必须是【chunk 总数】= bs*seq/P，不是 bs//P。
            #   用 bs//P 只会打乱前几个 chunk，permuted 效果被削弱，
            #   核心判据被严重低估（L-027）
            P = max(1, self._chunk(model))
            # ★ T 必须取【末维】(seq)，不能靠 dim() 判断。
            #   xb 是 [bs, seq] 的 token id 张量，dim()==2，
            #   若写成 `xb.shape[1] if xb.dim() > 2 else 1` 会得到 T=1
            #   ⇒ C = 64*1/32 = 2，只打乱 2 个 chunk（共 256 个），
            #   permuted 几乎等于 learned，核心判据 ≈ 0（见 L-033）
            T = xb.shape[-1]
            C = (xb.shape[0] * T) // P
            # 断言：perm 必须覆盖全部 chunk，否则核心判据失效
            if C < 8:
                print(f"[WARN] perm 长度 C={C} 过小（chunk 数应该是 "
                      f"bs*seq/P）。检查 P 与 seq 是否匹配", flush=True)
            for _ in range(n_perm):
                try:
                    perm = torch.randperm(C, generator=g, device=dev)
                except RuntimeError:
                    # generator device 仍不匹配时的兜底（牺牲可复现性）
                    perm = torch.randperm(C, device=dev)
                o, _ = model(xb, perm=perm)
                L["permuted"].append(_ce(o, yb).item())

        res = {f"loss_{k}": float(np.mean(v)) for k, v in L.items()}
        res["loss_permuted_sd"] = float(np.std(L["permuted"]))
        res["n_permute_repeats"] = int(n_perm)

        l_l, l_c, l_p = res["loss_learned"], res["loss_constant"], res["loss_permuted"]
        res["mix_gain"] = float(l_c - l_l)
        # 核心判据：permuted 与 learned 之差。>0 且 CI 下界>0 ⇒ 分工成立
        core = float(l_p - l_l)
        res["core_stat"] = core

        # 用 permuted 的重复采样做 t 检验（近似）
        sd = max(res["loss_permuted_sd"], 1e-12)
        n_eff = max(n_perm * n_batch, 2)
        se = sd / math.sqrt(n_eff)
        try:
            from scipy import stats as st
            tcrit = st.t.ppf(0.975, max(n_eff - 1, 1))
            res["core_stat_ci_lo"] = float(core - tcrit * se)
            res["core_stat_ci_hi"] = float(core + tcrit * se)
            res["cohens_dz"] = float(core / sd) if sd > 0 else 0.0
            res["p_core"] = float(2 * (1 - st.t.cdf(abs(core / se), max(n_eff - 1, 1))))
        except Exception:
            res["core_stat_ci_lo"] = res["core_stat_ci_hi"] = float(core)
            res["cohens_dz"] = 0.0
            res["p_core"] = 1.0
        return res

    @staticmethod
    def _unpack(batch, dev):
        if isinstance(batch, (list, tuple)):
            xb, yb = batch[0], batch[1]
        else:
            xb, yb = batch, batch
        return xb.to(dev), yb.to(dev)

    @staticmethod
    def _chunk(model):
        for layer in getattr(model, "layers", []):
            for c in layer:
                return getattr(c, "chunk_size", 1)
        return getattr(model, "chunk_size", 1)

    @staticmethod
    def _mix_ema(model):
        for layer in getattr(model, "layers", []):
            for c in layer:
                return getattr(c, "mix_ema", None)
        return getattr(model, "mix_ema", None)

    # ---------------------------------------------------------------- 结束
    def finalize(self, model=None, final=None, verdicts=None):
        f = dict(final or {})
        f["total_wall_sec"] = time.time() - self.t0
        f["sharp_warn_triggered"] = int(self.sharp_triggered)
        if model is not None:
            try:
                f["params_total"] = int(sum(p.numel() for p in model.parameters()))
            except Exception:
                pass
            st = self._model_stats(model)
            for k, dst in [("in_band", "in_band_final"),
                           ("mix_sharp", "mix_sharp_final"),
                           ("cos_mean", "cos_mean_final"),
                           ("combo_dim", "combo_dim"),
                           ("flops_ratio", "flops_ratio")]:
                if k in st:
                    f[dst] = float(st[k])
            f.setdefault("peak_mem_mb",
                         float(torch.cuda.max_memory_allocated() / 1024 ** 2)
                         if torch.cuda.is_available() else 0.0)
        return self.log.finalize(f, verdicts)

    # ---------------------------------------------------------------- 判据
    @staticmethod
    def auto_verdicts(final):
        """按预注册判据自动判定（见 EXPERIMENT_PROTOCOL.md H1~H7）。"""
        v = {}
        core = final.get("core_stat", None)
        if core is not None:
            ci_lo = final.get("core_stat_ci_lo", core)
            v["H4_分工真实存在"] = "pass" if (core > 0 and ci_lo > 0) else "fail"
        ib = final.get("in_band_final", None)
        if ib is not None:
            v["H1_连接带内占比"] = "pass" if ib >= 0.7 else "fail"
        wv = final.get("wiring_variance", None)
        if wv is not None:
            v["H2_跨样本方差"] = "pass" if wv > 0.05 else "fail"
        ms = final.get("mix_sharp_final", None)
        if ms is not None:
            v["H3_非onehot"] = "pass" if ms <= 0.7 else "warn"
        fr = final.get("flops_ratio", None)
        if fr is not None:
            v["H5_效率优于TopK"] = "pass" if fr < 1.0 else "fail"
        return v
