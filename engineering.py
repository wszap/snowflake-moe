# -*- coding: utf-8 -*-
"""Snowflake MoE 工程基础模块（清单一：1.1–1.10）

集中实现：
1.1 Checkpoint 保存机制（权重/优化器/配置/epoch/PPL -> checkpoints/）
1.2 多窗口评估（验证集 10 个固定 offset，每窗 2048 字符，均值±标准差）
1.3 设备匹配（新参数显式 device=model.device）
1.4 绝对路径清理（环境变量或相对路径，不再写死 Marvis 目录）
1.5 训练日志系统（JSON 日志：step/loss/lr/grad_norm/GPU温度/显存）
1.6 随机种子管理（统一 set_seed 覆盖 torch/numpy/random）
1.7 配置集中管理（单一 Config 类 + 可导入导出 JSON）
1.8 断点续训（从 checkpoint 恢复训练，含优化器状态/epoch）
1.9 内存/温度自动监控（GPU>80°C 或显存>7GB 自动暂停）
1.10 OOM 防御（try/except 捕获，自动降 batch size 重试）

独立模块：仅依赖 torch/numpy/json，不依赖其它项目文件。
"""
import json
import os
import random
import time

import numpy as np
import torch


# ================================================================ 1.6 种子管理
def set_seed(seed=2026):
    """统一随机种子：torch / numpy / random 全覆盖。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================ 1.7 配置集中
class Config:
    """集中配置：所有超参收敛到单一对象，可 JSON 导入导出。

    用法：cfg = Config.load(path) 或 Config(**defaults)；
    修改一处即全局生效（各训练/评估函数只读 cfg 字段）。
    """
    _FIELDS = dict(
        # 数据
        data_path=None, vocab_size=0, seq_len=64, batch_size=16, accum=8,
        # 模型
        d=64, h=128, E=4, S=1, L=2, topk=2,
        n_organelles=20, n_memory=64, topk_organelle=4,
        # 训练
        epochs=5, lr=3e-4, warmup_frac=0.05, clip_norm=1.0,
        seed=2026, eval_windows=10, eval_win_len=2048,
        # 红线
        gpu_max_temp=80, vram_max_gb=7.0,
        # 路径（1.4：默认相对/环境变量）
        checkpoint_dir=None, log_dir=None,
    )

    def __init__(self, **kwargs):
        for k, v in self._FIELDS.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            if k not in self._FIELDS:
                raise KeyError(f"未知配置项: {k}")
            setattr(self, k, v)
        # 默认路径：优先环境变量，其次相对当前文件（1.4 可移植）
        base = os.environ.get("SNOWFLAKE_BASE", os.getcwd())
        if self.checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(base, "checkpoints")
        if self.log_dir is None:
            self.log_dir = os.path.join(base, "logs")

    def to_dict(self):
        return {k: getattr(self, k) for k in self._FIELDS}

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(**d)


# ================================================================ 1.1 / 1.8 Checkpoint
class CheckpointManager:
    """Checkpoint 保存/恢复：权重 + 优化器 + epoch + 最佳 PPL + 配置。

    save(model, opt, epoch, ppl, tag) -> checkpoints/{tag}_ep{epoch}_ppl{ppl:.3f}.pt
    resume(model, opt, ckpt_path) -> 返回 (epoch, best_ppl)，恢复优化器状态。
    """

    def __init__(self, cfg: Config, tag="snowflake"):
        self.cfg = cfg
        self.tag = tag
        self.dir = cfg.checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)

    def save(self, model, opt, epoch, ppl=None, extra=None):
        path = os.path.join(
            self.dir,
            f"{self.tag}_ep{epoch:02d}_ppl{float(ppl):.3f}.pt" if ppl is not None
            else f"{self.tag}_ep{epoch:02d}.pt")
        payload = {
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict() if opt is not None else None,
            "epoch": epoch,
            "ppl": ppl,
            "config": self.cfg.to_dict(),
            "extra": extra or {},
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(payload, path)
        return path

    def resume(self, model, opt=None, ckpt_path=None, device=None):
        """恢复模型与优化器状态。返回 (epoch, ppl)。找不到则返回 (0, None)。"""
        if ckpt_path is None:
            cands = sorted(
                [f for f in os.listdir(self.dir) if f.startswith(self.tag)],
                key=lambda f: os.path.getmtime(os.path.join(self.dir, f)),
                reverse=True)
            if not cands:
                return 0, None
            ckpt_path = os.path.join(self.dir, cands[0])
        payload = torch.load(ckpt_path, map_location=device or "cpu",
                             weights_only=False)
        model.load_state_dict(payload["model_state"])
        if opt is not None and payload.get("opt_state") is not None:
            opt.load_state_dict(payload["opt_state"])
        return int(payload.get("epoch", 0)), payload.get("ppl")


# ================================================================ 1.2 多窗口评估
def eval_multi_window(model, val_ids, vocab_size, seq_len=64, batch_size=16,
                      n_windows=10, win_len=2048, device="cpu"):
    """验证集 10 个固定 offset，每窗 win_len 字符，逐窗连续分块评估。

    返回 (mean_ce, mean_ppl, std_ppl, ppl_list)。同模型两次评估差异 < 0.5。
    """
    model.eval()
    n = val_ids.numel()
    n_windows = min(n_windows, max(1, n // win_len))
    stride = max(1, (n - win_len) // n_windows) if n_windows > 1 else 0
    offsets = [min(i * stride, n - win_len) for i in range(n_windows)]
    k = batch_size * seq_len
    ppl_list = []
    with torch.no_grad():
        for off in offsets:
            seg = val_ids[off:off + win_len]
            total_ce, total_n = 0.0, 0
            s = 0
            while s + k + 1 <= seg.numel():
                x = seg[s:s + k].view(batch_size, seq_len).to(device)
                y = seg[s + 1:s + k + 1].view(batch_size, seq_len).to(device)
                logits, _ = model(x)
                ce = torch.nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)).item()
                total_ce += ce * k
                total_n += k
                s += k
            if total_n > 0:
                ppl_list.append(float(np.exp(total_ce / total_n)))
    model.train()
    if not ppl_list:
        return 0.0, 0.0, 0.0, []
    p = np.array(ppl_list)
    return float(p.mean()), float(np.exp(np.log(p).mean())), float(p.std()), ppl_list


# ================================================================ 1.5 训练日志
class JsonLogger:
    """JSON 日志：每 step 记录 loss/lr/grad_norm/GPU温度/显存，可绘曲线。"""

    def __init__(self, log_dir, name="train"):
        self.path = os.path.join(log_dir, f"{name}.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")

    def log(self, step, **kwargs):
        rec = dict(step=step, ts=time.strftime("%Y-%m-%d %H:%M:%S"))
        rec.update(kwargs)
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ================================================================ 1.9 温度/显存监控
def gpu_status(device="cuda"):
    """返回 (温度°C, 显存GB)。CPU 或无 GPU 返回 (None, 0)。"""
    if device != "cuda" or not torch.cuda.is_available():
        return None, 0.0
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used",
             "--format=csv,noheader,nounits"], text=True)
        temp_s, mem_s = out.strip().splitlines()[0].split(",")
        return int(temp_s.strip()), float(mem_s.strip()) / 1024.0
    except Exception:  # noqa: BLE001
        try:
            temp = torch.cuda.temperature() if hasattr(torch.cuda, "temperature") else 0
            mem = torch.cuda.memory_allocated() / 1024 ** 3
            return int(temp), float(mem)
        except Exception:  # noqa: BLE001
            return None, 0.0


def guard_redline(cfg: Config, wait_s=300):
    """红线监控：GPU>80°C 或显存>7GB 自动暂停。返回 (temp, vram)。"""
    temp, vram = gpu_status("cuda" if torch.cuda.is_available() else "cpu")
    while temp is not None and temp >= cfg.gpu_max_temp:
        print(f"[GUARD] GPU {temp}°C >= {cfg.gpu_max_temp}°C，暂停 {wait_s}s", flush=True)
        time.sleep(wait_s)
        temp, vram = gpu_status("cuda" if torch.cuda.is_available() else "cpu")
    while vram and vram >= cfg.vram_max_gb:
        print(f"[GUARD] 显存 {vram:.1f}GB >= {cfg.vram_max_gb}GB，暂停 {wait_s}s", flush=True)
        time.sleep(wait_s)
        temp, vram = gpu_status("cuda" if torch.cuda.is_available() else "cpu")
    return temp, vram


# ================================================================ 1.10 OOM 防御
def with_oom_retry(fn, batch_size, min_batch=2, scale=0.5, max_retry=3):
    """执行 fn(batch_size)；OOM 时自动降 batch 重试。

    fn 需返回 (result, used_batch)。OOM（RuntimeError 含 out of memory）
    时按 scale 缩小 batch，最多 max_retry 次；仍失败则抛原异常。
    """
    cur = batch_size
    last_err = None
    for _ in range(max_retry + 1):
        try:
            return fn(cur), cur
        except RuntimeError as e:
            last_err = e
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower() and "memory" in str(e).lower():
                nxt = max(min_batch, int(cur * scale))
                if nxt >= cur:
                    raise
                print(f"[OOM] batch {cur} -> {nxt} 重试", flush=True)
                cur = nxt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise
    raise last_err
