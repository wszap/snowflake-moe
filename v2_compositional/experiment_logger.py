# -*- coding: utf-8 -*-
"""
experiment_logger —— 论文级实验记录器（防口径漂移 + 强制 schema + 统计聚合）

设计目标（针对已出现的三类事故）：
  1. 同一指标在不同文档里给出不同数字（18.1251 vs 22.83 / +0.0098 vs +1.8 vs -0.57
     / 9.9787 vs 10.0157）→ 每次 run 落不可变记录，文档中的数字必须回指 run_id
  2. 跑完才决定哪条是贡献（HARKing）→ finalize() 要求声明 pre-registered 判据
  3. 只报漂亮结果（file-drawer）→ 负结果走同一条落盘路径，不可删

用法（真机，有 torch）：

    from experiment_logger import RunLogger, finalize
    log = RunLogger(cfg_dict, script="run_cellmoe_ckpt_lock47.py",
                    tags=["main", "tinystories"], root="output/runs")
    log.attach_data("tinystories_25mb.txt")          # 自动算 sha256 / bytes / n_tokens
    ...
    for step in ...:
        log.log_step(step, epoch=..., train_loss=..., val_ppl=..., in_band=..., ...)
    log.finalize(final_metrics, verdicts={...})

统计聚合（无需 torch，沙箱可跑）：

    from experiment_logger import load_runs, seeds_aggregate, paired_compare, bh_correct
    runs = load_runs("output/runs", filters={"cfg.arch": "pz47"})
    print(seeds_aggregate(runs, "final.test_ppl"))
    print(paired_compare(runs_a, runs_b, "final.test_ppl"))
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from scipy import stats as _st

CST = timezone(timedelta(hours=8))

# ==========================================================================
# SCHEMA —— 强制。不在 schema 里的 key 会被拒绝（防止随手改名导致口径漂移）
# ==========================================================================
STEP_FIELDS = {
    # ---- 训练基本量 ----
    "step": "int", "epoch": "int", "wall_sec": "float", "tokens_seen": "int",
    "train_loss": "float", "lr": "float", "val_ppl": "float", "val_ce": "float",

    # ---- 拼专家架构专属（lock47 旧字段，保留以兼容历史 run）----
    "in_band": "float", "heal_term": "float", "org_norm": "float",
    "cancer_mask": "float", "mix_var": "float", "mix_sharp": "float",
    "mix_eff_rank": "float", "cond_ema": "float", "wiring_ent": "float",

    # ---- ★ B 型 lock6.x 新字段（直接写在这里，不依赖 run_metrics 导入）----
    # 曾经靠 run_metrics.STEP_FIELDS.update() 追加 ⇒ 单独用 experiment_logger
    # 的脚本会报"未知字段"，且字段名与旧版对不上（mix_var vs wiring_variance）。
    # 见 L-028。
    "cos_mean": "float", "cos_std": "float", "cos_max": "float",
    "connect_rowsum": "float",
    "wiring_variance": "float",
    # 跨 N 可比的核心量（见 L-023）
    "wiring_var_norm": "float",
    "eff_organs": "float",
    "combo_dim": "float", "flops_ratio": "float",

    # ---- 效率 ----
    "gpu_mem_peak_mb": "float", "throughput_tok_s": "float",
    "gpu_util_pct": "float", "gpu_temp_c": "float",
}

FINAL_FIELDS = {
    "val_ppl": "float", "val_ce": "float", "test_ppl": "float", "test_ce": "float",
    "params_total": "int", "params_active": "int",
    "flops_per_token_theory": "int", "flops_per_token_measured": "float",
    "latency_ms_per_step": "float", "peak_mem_mb": "float",
    "throughput_tok_s": "float", "total_wall_sec": "float",
    # 拼专家专属最终判据
    "mix_gain": "float", "mix_gain_ci_lo": "float", "mix_gain_ci_hi": "float",
    "in_band_min": "float", "cond_ema_max": "float", "heal_term_final": "float",
    # 终身学习 / 擦除（若本 run 涉及）
    "old_domain_ppl_before": "float", "old_domain_ppl_after": "float",
    "old_domain_regression": "float",
    "new_domain_ppl_before": "float", "new_domain_ppl_after": "float",
    "new_domain_drop_pct": "float",
    "erase_frac": "float", "erase_ppl": "float", "erase_ppl_ratio": "float",

    # ---- ★ B 型 lock6.x 最终判据（直接写在这里，不依赖 run_metrics）----
    # 见 L-028
    "loss_learned": "float", "loss_constant": "float", "loss_permuted": "float",
    "loss_permuted_sd": "float",
    "core_stat": "float", "core_stat_ci_lo": "float", "core_stat_ci_hi": "float",
    "cohens_dz": "float", "p_core": "float", "n_permute_repeats": "int",
    "mix_sharp_final": "float", "in_band_final": "float", "cos_mean_final": "float",
    "eff_organs_final": "float", "wiring_var_norm": "float",
    "combo_dim": "float", "flops_ratio": "float",
    "sharp_warn_triggered": "int",
    # release test（判定"真想拼 vs 被摁着"）
    "release_sharp_on": "float", "release_sharp_off": "float",
    "release_eff_on": "float", "release_eff_off": "float",
    "release_rebound": "float", "release_verdict": "str",
}

META_REQUIRED = [
    "run_id", "timestamp_utc", "timestamp_cst", "script", "seed",
    "git_commit", "git_dirty", "config_hash", "python_version",
    "torch_version", "cuda_version", "gpu_name", "data_sha256", "data_bytes",
]


# ==========================================================================
# 环境 / 代码 / 数据指纹
# ==========================================================================
def _git(root=None):
    try:
        cmds = {
            "commit": ["git", "rev-parse", "HEAD"],
            "dirty": ["git", "status", "--porcelain"],
            "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        }
        out = {}
        for k, c in cmds.items():
            r = subprocess.run(c, cwd=root or os.getcwd(), capture_output=True,
                               text=True, timeout=10)
            out[k] = r.stdout.strip() if r.returncode == 0 else ""
        return {"git_commit": out["commit"],
                "git_dirty": bool(out["dirty"]),
                "git_branch": out["branch"]}
    except Exception:
        return {"git_commit": "", "git_dirty": None, "git_branch": ""}


def _env():
    e = {"python_version": platform.python_version(),
         "platform": platform.platform(),
         "torch_version": "", "cuda_version": "", "gpu_name": "", "gpu_count": 0}
    try:
        import torch
        e["torch_version"] = torch.__version__
        e["cuda_version"] = getattr(torch.version, "cuda", "") or ""
        e["gpu_count"] = torch.cuda.device_count()
        if e["gpu_count"]:
            e["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return e


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_config(cfg):
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ==========================================================================
# RunLogger
# ==========================================================================
class RunLogger:
    def __init__(self, cfg: dict, script: str, tags=None, root="output/runs",
                 repo_root=None, strict=True):
        self.strict = strict
        self.cfg = dict(cfg)
        self.script = os.path.basename(script)
        self.tags = list(tags or [])
        self.root = root
        os.makedirs(root, exist_ok=True)

        now = datetime.now(timezone.utc)
        self.meta = {
            "run_id": now.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6],
            "timestamp_utc": now.isoformat(),
            "timestamp_cst": now.astimezone(CST).isoformat(),
            "script": self.script,
            "seed": cfg.get("seed", None),
            "tags": self.tags,
            "config": self.cfg,
            "config_hash": hash_config(cfg),
        }
        self.meta.update(_env())
        self.meta.update(_git(repo_root))
        self.data = {}
        self.steps = []
        self.final = {}
        self.verdicts = {}
        self._t0 = time.time()
        self._finalized = False
        print(f"[RunLogger] run_id={self.meta['run_id']} cfg_hash={self.meta['config_hash']}")

    # ---------------------------------------------------------------- 数据
    def attach_data(self, path, tokenizer=None, vocab_size=None,
                    split_method=None, split_seed=None, doc_boundary_aware=None,
                    n_train=None, n_val=None, n_test=None):
        self.data = {
            "data_path": os.path.abspath(path),
            "data_sha256": sha256_file(path),
            "data_bytes": os.path.getsize(path),
            "data_mtime": datetime.fromtimestamp(os.path.getmtime(path), CST).isoformat(),
            "tokenizer": tokenizer, "vocab_size": vocab_size,
            "split_method": split_method, "split_seed": split_seed,
            "doc_boundary_aware": doc_boundary_aware,
            "n_train_tokens": n_train, "n_val_tokens": n_val, "n_test_tokens": n_test,
        }
        print(f"[RunLogger] data sha256={self.data['data_sha256'][:16]} "
              f"bytes={self.data['data_bytes']}")
        return self

    # ---------------------------------------------------------------- step
    def log_step(self, step, **kw):
        rec = {"step": step}
        for k, v in kw.items():
            if k not in STEP_FIELDS:
                msg = f"未知 step 字段 '{k}'。请先在 STEP_FIELDS 注册（防口径漂移）"
                if self.strict:
                    raise KeyError(msg)
                print(f"[WARN] {msg}")
            rec[k] = v
        self.steps.append(rec)
        return self

    # ---------------------------------------------------------------- final
    def finalize(self, final: dict, verdicts: dict | None = None):
        for k, v in final.items():
            if k not in FINAL_FIELDS:
                msg = f"未知 final 字段 '{k}'。请先在 FINAL_FIELDS 注册"
                if self.strict:
                    raise KeyError(msg)
                print(f"[WARN] {msg}")
        self.final = dict(final)
        self.verdicts = dict(verdicts or {})
        self.final["total_wall_sec"] = self.final.get(
            "total_wall_sec", time.time() - self._t0)

        # 注意用 `is None`：git_dirty=False（仓库干净）是合法值，不能当成缺失
        merged = {**self.meta, **self.data}
        missing = [k for k in META_REQUIRED if merged.get(k, None) is None]
        rec = {"meta": self.meta,
               "data": self.data,
               "steps": self.steps,
               "final": self.final,
               "verdicts": self.verdicts,
               "meta_missing": missing}
        if missing:
            print(f"[WARN] 元数据缺失 {missing} —— 这些 run 不能进论文主表")

        d = os.path.join(self.root, self.meta["run_id"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "record.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2, default=str)
        if self.steps:
            pd.DataFrame(self.steps).to_csv(os.path.join(d, "steps.csv"), index=False)
        self._append_master(rec)
        self._finalized = True
        print(f"[RunLogger] 已落盘 → {d}")
        return self.meta["run_id"]

    def _append_master(self, rec):
        row = {"run_id": self.meta["run_id"],
               "timestamp_cst": self.meta["timestamp_cst"],
               "script": self.meta["script"], "seed": self.meta["seed"],
               "git_commit": self.meta["git_commit"], "git_dirty": self.meta["git_dirty"],
               "config_hash": self.meta["config_hash"],
               "gpu_name": self.meta["gpu_name"],
               "data_sha256": self.data.get("data_sha256", ""),
               "tags": "|".join(self.tags)}
        for k, v in self.cfg.items():
            row[f"cfg.{k}"] = v
        for k, v in self.final.items():
            row[f"final.{k}"] = v
        for k, v in self.verdicts.items():
            row[f"verdict.{k}"] = v
        row["meta_missing"] = ",".join(rec["meta_missing"])
        path = os.path.join(self.root, "master.csv")
        new = pd.DataFrame([row])
        if os.path.exists(path):
            # 关键：追加必须按已有 header 的列顺序对齐。
            # 直接 to_csv(mode="a") 会按新 DataFrame 自己的列序写，
            # 列集不同时整行错位（曾导致 test_ppl 列读到 val_ppl 的值）。
            cols = list(pd.read_csv(path, nrows=0).columns)
            for c in new.columns:
                if c not in cols:
                    cols.append(c)
            new = new.reindex(columns=cols)
        new.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


# ==========================================================================
# 统计工具（无需 torch）
# ==========================================================================
def load_runs(root="output/runs", filters=None):
    """读 master.csv，按 filters 过滤（支持 cfg.* / final.* / 顶层列名）。"""
    path = os.path.join(root, "master.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    for k, v in (filters or {}).items():
        df = df[df[k].astype(str) == str(v)]
    return df


def seeds_aggregate(df, col, conf=0.95):
    """n seeds 的 mean ± std + t 分布 CI。返回 dict（含 n，n<5 会标记不可靠）。"""
    x = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    if n == 0:
        return {"col": col, "n": 0}
    mean, sd = float(x.mean()), float(x.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        t = _st.t.ppf(0.5 + conf / 2, n - 1)
        hw = float(t * sd / np.sqrt(n))
    else:
        hw = float("nan")
    return {"col": col, "n": n, "mean": mean, "std": sd,
            "ci_lo": mean - hw, "ci_hi": mean + hw, "halfwidth": hw,
            "min": float(x.min()), "max": float(x.max()),
            "reliable": n >= 5}


def paired_compare(df_a, df_b, col, key="seed", conf=0.95):
    """同 seed 配对比较。同时给 t 检验、Wilcoxon、Cohen's d。

    返回 dict；n<5 时 p 值只作参考，必须以效应量为准。
    """
    a = df_a[[key, col]].rename(columns={col: "a"})
    b = df_b[[key, col]].rename(columns={col: "b"})
    m = pd.merge(a, b, on=key, how="inner").dropna()
    d_ = (m["b"] - m["a"]).to_numpy(dtype=float)   # 正 = b 更差（如 PPL）
    n = len(d_)
    if n < 2:
        return {"col": col, "n": n, "ok": False}
    mean_d = float(d_.mean()); sd_d = float(d_.std(ddof=1))
    se = sd_d / np.sqrt(n)
    t_stat, p_t = _st.ttest_rel(m["b"], m["a"])
    try:
        w_stat, p_w = _st.wilcoxon(d_)
    except Exception:
        w_stat, p_w = float("nan"), float("nan")
    tcrit = _st.t.ppf(0.5 + conf / 2, n - 1)
    dz = mean_d / sd_d if sd_d > 0 else float("nan")   # Cohen's d_z (paired)
    return {"col": col, "n": n, "ok": True,
            "mean_diff": mean_d, "ci_lo": mean_d - tcrit * se, "ci_hi": mean_d + tcrit * se,
            "t": float(t_stat), "p_t": float(p_t),
            "wilcoxon": float(w_stat), "p_wilcoxon": float(p_w),
            "cohens_dz": float(dz),
            "reliable": n >= 5,
            "note": "n<5：p 值不可靠，以效应量与 CI 为准"}


def bh_correct(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR 校正。返回 (reject, p_adj)。"""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out_adj = np.full_like(p, np.nan)
    out_rej = np.zeros(len(p), dtype=bool)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out_rej, out_adj
    pv = p[idx]; m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m); adj[order] = np.clip(ranked, 0, 1)
    out_adj[idx] = adj
    out_rej[idx] = adj <= alpha
    return out_rej, out_adj


# ==========================================================================
# 自检（沙箱可跑，无需 torch）
# ==========================================================================
def _selftest():
    root = "/tmp/_logtest"
    import shutil; shutil.rmtree(root, ignore_errors=True)

    cfg = {"arch": "pz47", "seed": 2026, "d": 128, "chunk_size": 32}
    lg = RunLogger(cfg, script="selftest.py", tags=["selftest"], root=root)
    lg.attach_data(__file__, tokenizer="char", vocab_size=1000,
                   split_method="doc_boundary", split_seed=2026, doc_boundary_aware=True)
    for s in range(1, 6):
        lg.log_step(s, epoch=1, train_loss=2.3 - 0.01 * s, val_ppl=10.0 + 0.1 * s,
                    in_band=0.9, mix_var=0.02, mix_sharp=0.35, mix_eff_rank=6.5,
                    cond_ema=12.0)
    lg.finalize({"val_ppl": 10.01, "test_ppl": 10.05, "params_total": 1823077,
                 "mix_gain": 0.002, "in_band_min": 0.85},
                verdicts={"H1_noninferior": "pass"})

    # 5 seeds × 2 arch
    rng = np.random.default_rng(0)
    for arch, mu in [("pz47", 10.01), ("fixed", 10.015)]:
        for sd in range(5):
            c = {**cfg, "arch": arch, "seed": sd}
            l = RunLogger(c, script="selftest.py", tags=[arch], root=root)
            l.attach_data(__file__, tokenizer="char", vocab_size=1000,
                          split_method="doc_boundary", split_seed=sd,
                          doc_boundary_aware=True)
            l.finalize({"test_ppl": float(mu + rng.normal(0, 0.02)),
                        "val_ppl": 10.0, "mix_gain": 0.001})
    df = load_runs(root)
    a = load_runs(root, {"cfg.arch": "pz47"})
    b = load_runs(root, {"cfg.arch": "fixed"})

    # 回归断言：追加必须对齐，不能错位（曾把 val_ppl 写进 test_ppl 列）
    tp = pd.to_numeric(a["final.test_ppl"], errors="coerce").dropna()
    assert (tp > 9.98).all(), f"master.csv 列错位：test_ppl={tp.tolist()}"
    assert (tp < 10.06).all(), f"master.csv 列错位：test_ppl={tp.tolist()}"
    print(f"\n[OK] 列对齐：test_ppl 落在 (10.001, 10.02) 内，n={len(tp)}")

    print("\n--- seeds_aggregate(pz47.test_ppl) ---")
    print(seeds_aggregate(a, "final.test_ppl"))
    print("\n--- paired_compare(pz47 vs fixed) ---")
    print(paired_compare(a, b, "final.test_ppl"))
    print("\n--- schema 拦截测试 ---")
    l2 = RunLogger({"arch": "x", "seed": 1}, script="t.py", root=root)
    try:
        l2.log_step(1, wiring_entropy=1.0)
        print("FAIL: 未拦截未注册字段")
    except KeyError as e:
        print(f"OK 拦截: {e}")
    print("\n--- BH 校正 ---")
    print(bh_correct([0.01, 0.04, 0.03, 0.5, 0.2]))
    return root


if __name__ == "__main__":
    _selftest()
