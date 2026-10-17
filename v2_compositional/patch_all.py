#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_all.py —— 一次性修复 run_metrics.py 的 6 个 bug

★ 必须在 10 seeds 跑完（ALL_DONE）之后执行
   训练途中改代码会让前后 seed 口径不一致，统计检验作废

用法:
    python patch_all.py --check    # 只检查，不改
    python patch_all.py --apply    # 应用（自动备份 .bak_patchall）
    python patch_all.py --verify   # 验证补丁生效
"""

import io
import os
import sys
import shutil

TARGETS = ["run_metrics.py", "snowflake_B.py", "train_lock60.py"]


# ════════════════════════════════════════════════════════════════
# Bug 1: _model_stats.key_map 缺 eff_organs / wiring_var_norm
# ════════════════════════════════════════════════════════════════
B1_OLD = '''        key_map = {
            "in_band": "in_band", "cos_mean": "cos_mean", "cos_std": "cos_std",
            "cos_max": "cos_max", "connect_rowsum": "connect_rowsum",
            "wiring_ent": "wiring_ent", "wiring_variance": "wiring_variance",
            "mix_sharp": "mix_sharp",
        }'''
B1_NEW = '''        key_map = {
            "in_band": "in_band", "cos_mean": "cos_mean", "cos_std": "cos_std",
            "cos_max": "cos_max", "connect_rowsum": "connect_rowsum",
            "wiring_ent": "wiring_ent", "wiring_variance": "wiring_variance",
            "mix_sharp": "mix_sharp",
            # ★ Bug1: 这两个是【核心量】，之前漏了 → eff 从未落盘。
            #   eff_organs = 1/Σw²，跨 N 可比（L-023）
            #   wiring_var_norm = wvar/var_max，解决 N=384 下 0.05 不可达
            "eff_organs": "eff_organs",
            "wiring_var_norm": "wiring_var_norm",
        }'''

# ════════════════════════════════════════════════════════════════
# Bug 2: finalize 映射缺 eff_organs_final / wiring_variance / var_norm
# ════════════════════════════════════════════════════════════════
B2_OLD = '''            for k, dst in [("in_band", "in_band_final"),
                           ("mix_sharp", "mix_sharp_final"),
                           ("cos_mean", "cos_mean_final"),
                           ("combo_dim", "combo_dim"),
                           ("flops_ratio", "flops_ratio")]:
                if k in st:
                    f[dst] = float(st[k])'''
B2_NEW = '''            for k, dst in [("in_band", "in_band_final"),
                           ("mix_sharp", "mix_sharp_final"),
                           ("cos_mean", "cos_mean_final"),
                           ("combo_dim", "combo_dim"),
                           ("flops_ratio", "flops_ratio"),
                           # ★ Bug2: eff / wiring 方差从未进 final
                           ("eff_organs", "eff_organs_final"),
                           ("wiring_variance", "wiring_variance"),
                           ("wiring_var_norm", "wiring_var_norm")]:
                if k in st:
                    f[dst] = float(st[k])'''

# ════════════════════════════════════════════════════════════════
# Bug 3+6: auto_verdicts 的 H2 用不可达阈值 + 键名不匹配
# ════════════════════════════════════════════════════════════════
B3_OLD = '''        wv = final.get("wiring_variance", None)
        if wv is not None:
            v["H2_跨样本方差"] = "pass" if wv > 0.05 else "fail"'''
B3_NEW = '''        # ★ Bug3: 原用 wiring_variance + 0.05 阈值。
        #   N=384 下 wvar 的理论上限仅 0.0026，0.05 数学上不可达（L-023）。
        #   ⇒ 改用 wiring_var_norm = wvar/var_max，阈值 0.05 才有意义。
        # ★ Bug6: 原代码 get("wiring_variance")，但注册的是 wiring_var_norm，
        #   键名不一致 ⇒ 即使填了值也读不到，H2 被静默跳过。
        wv = final.get("wiring_var_norm", None)
        if wv is None:                       # 兼容老数据：现场归一化
            raw = final.get("wiring_variance", None)
            n = final.get("n_organelles", 384)
            if raw is not None and n:
                _m = 1.0 / n
                vmax = (1.0 / n) * (1 - _m) ** 2 + ((n - 1) / n) * _m ** 2
                wv = raw / vmax if vmax > 0 else raw
        if wv is not None:
            v["H2_跨样本方差"] = "pass" if wv > 0.05 else "fail"

        # ★ 新增 H2b：有效器官数判据（比方差更直观，且跨 N 可比）
        ef = final.get("eff_organs_final", None)
        if ef is not None:
            # one-hot ≈ 1；真组合应远大于 1
            v["H2b_有效器官数"] = "pass" if ef >= 8 else (
                "warn" if ef >= 2 else "fail")'''

# ════════════════════════════════════════════════════════════════
# Bug 5: sharp 告警计数采样遗漏 —— 从模型侧读累计值
# ════════════════════════════════════════════════════════════════
B5_OLD = '''        f["total_wall_sec"] = time.time() - self.t0
        f["sharp_warn_triggered"] = int(self.sharp_triggered)'''
B5_NEW = '''        f["total_wall_sec"] = time.time() - self.t0
        # ★ Bug5: 只在 log_step 采样点（每 100 步）检查 sharp，
        #   实测 step 10~90 sharp 高达 0.75~0.97，但采样点在 step0(0.008)/
        #   step100(0.178) 全部错过 ⇒ 计数恒为 0。
        #   改为：优先读模型侧的【逐步累计】计数，采样计数仅作兜底。
        model_cnt = 0
        if model is not None:
            for layer in getattr(model, "layers", []):
                for c in layer:
                    model_cnt += int(getattr(c, "sharp_warn_count", 0) or 0)
        f["sharp_warn_triggered"] = int(max(self.sharp_triggered, model_cnt))
        f["sharp_warn_sampled"] = int(self.sharp_triggered)   # 诊断用
        f["sharp_warn_total"] = int(model_cnt)                # 诊断用'''


# ════════════════════════════════════════════════════════════════
# Bug5b: snowflake_B.py 缺逐步 sharp 告警累计计数器
#   （run_metrics 只能每 100 步采样，必须有逐步计数才不漏）
# ════════════════════════════════════════════════════════════════
B5B_OLD = """        self.log_sharp = log_sharp"""
B5B_NEW = """        self.log_sharp = log_sharp
        # ★ Bug5b: 逐步累计的 one-hot 告警计数。
        #   run_metrics 每 100 步才采样一次，会漏掉 step10~90 的
        #   sharp=0.75~0.97（采样点在 step0/100 恰好都低于阈值）。
        #   这里逐步累计，finalize 时读取，保证计数不失真。
        self.sharp_warn_count = 0"""

B5B2_OLD = """        if self.log_sharp and float(sharp) > self.sharp_warn:"""
B5B2_NEW = """        if float(sharp) > self.sharp_warn:
            self.sharp_warn_count += 1
        if self.log_sharp and float(sharp) > self.sharp_warn:"""

# ════════════════════════════════════════════════════════════════
# Bug 9【最重要】: core_stat 的 CI / p / d_z 统计错误
#   ① 伪重复(pseudo-replication)：n_eff = n_perm*n_batch = 20*50 = 1000，
#      但同一 batch 内的 20 个 perm 共享同一组 xb/yb，不是独立样本
#      ⇒ se 被缩小 √n_perm ≈ 4.5 倍，CI 虚窄，p 虚假地小
#   ② 非配对：permuted 与 learned 在同一 batch 上测得，却当成两个独立样本
#      忽略了 learned 自身的变异，且丢掉了配对信息
#   实测印证：sd=0.0256, n_eff=1000 ⇒ CI 半宽 0.0016，
#      与你打印的 CI=[+0.7823,+0.7856]（半宽 0.0016）完全吻合 ⇒ 确实是旧算法
#   修正：先在 batch 内聚合 permuted，再与 learned 求配对差，做配对 t 检验
# ════════════════════════════════════════════════════════════════
B9_OLD = '''        # 用 permuted 的重复采样做 t 检验（近似）
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
        return res'''

B9_NEW = '''        # ★ Bug9: 下面这段是新写的【配对】检验，替换掉原来的伪重复估计。
        #   core 本身（两个 loss 之差）不变，变的是 se/CI/p/d_z —— 全部变保守。
        #   L["permuted"] 的 append 顺序是 batch 外层 / perm 内层，
        #   所以可以按 n_perm 切段还原出每个 batch 的 permuted 均值。
        _P = list(L["permuted"]); _Lr = list(L["learned"])
        _nb = min(len(_Lr), len(_P) // max(n_perm, 1))
        _diffs = []
        for i in range(_nb):
            seg = _P[i * n_perm:(i + 1) * n_perm]
            if seg:
                # 配对差：同一 batch 上的 (permuted 均值 − learned)
                _diffs.append(float(np.mean(seg)) - float(_Lr[i]))
        if len(_diffs) > 1:
            _d = np.asarray(_diffs)
            sd_d = float(np.std(_d, ddof=1))
            se = sd_d / math.sqrt(len(_d))
            n_eff = len(_d)
            res["core_stat_paired_sd"] = sd_d      # 配对差的 sd（新）
            res["n_paired_batches"] = int(n_eff)   # 有效样本数（新）
        else:
            # 兜底：退化为按 batch 数（不乘 n_perm）的保守估计
            sd_d = max(res["loss_permuted_sd"], 1e-12)
            n_eff = max(n_batch, 2)
            se = sd_d / math.sqrt(n_eff)
        res["n_eff_used"] = int(n_eff)             # 诊断：旧代码会是 1000
        try:
            from scipy import stats as st
            tcrit = st.t.ppf(0.975, max(n_eff - 1, 1))
            res["core_stat_ci_lo"] = float(core - tcrit * se)
            res["core_stat_ci_hi"] = float(core + tcrit * se)
            # Cohen's d_z 用配对差的 sd 才对（原来是 permuted 单样本的 sd）
            res["cohens_dz"] = float(core / sd_d) if sd_d > 0 else 0.0
            res["p_core"] = float(2 * (1 - st.t.cdf(abs(core / se), max(n_eff - 1, 1))))
        except Exception:
            res["core_stat_ci_lo"] = res["core_stat_ci_hi"] = float(core)
            res["cohens_dz"] = 0.0
            res["p_core"] = 1.0
        return res'''


# ════════════════════════════════════════════════════════════════
# Bug4: 早期采样太疏（每 100 步），前 500 步剧变全漏
# ════════════════════════════════════════════════════════════════
B4_OLD = """            if step % 100 == 0:
                dg = model.agg_diag()"""
B4_NEW = """            # ★ Bug4: 原每 100 步才采一次，但 step0~100 内
            #   eff 从 335 骤降到 113（早期剧变），论文要的
            #   收敛曲线前段是空的。改为前 500 步每 10 步采一次。
            _dense = (step < 500 and step % 10 == 0)
            if step % 100 == 0 or _dense:
                dg = model.agg_diag()"""


# ════════════════════════════════════════════════════════════════
# Bug 10: FINAL_FIELDS 未注册 Bug9/Bug5 新增字段
#   （漏注册会让 log.finalize 直接抛错 —— 前面注释里已踩过同样的坑）
# ════════════════════════════════════════════════════════════════
B10_OLD = '''    "cohens_dz": "float", "p_core": "float", "n_permute_repeats": "int",'''
B10_NEW = '''    "cohens_dz": "float", "p_core": "float", "n_permute_repeats": "int",
    # ★ Bug10: Bug9 新增的配对检验字段，漏注册会让 finalize 抛错
    "core_stat_paired_sd": "float", "n_paired_batches": "int",
    "n_eff_used": "int",
    # ★ Bug10: Bug5 新增的告警计数诊断字段
    "sharp_warn_sampled": "int", "sharp_warn_total": "int",
    "wiring_variance": "float", "n_organelles": "int",'''

PATCHES = [
    ("Bug9【重要】core_stat 伪重复+非配对 → 配对检验", "run_metrics.py", B9_OLD, B9_NEW),
    ("Bug10 注册 Bug9/Bug5 新增字段", "run_metrics.py", B10_OLD, B10_NEW),
    ("Bug1 key_map 缺 eff_organs/wiring_var_norm", "run_metrics.py", B1_OLD, B1_NEW),
    ("Bug2 finalize 缺 eff/variance 映射", "run_metrics.py", B2_OLD, B2_NEW),
    ("Bug3+6 H2 不可达阈值 + 键名不匹配", "run_metrics.py", B3_OLD, B3_NEW),
    ("Bug5 sharp 告警计数采样遗漏", "run_metrics.py", B5_OLD, B5_NEW),
    ("Bug5b 模型侧逐步告警计数器", "snowflake_B.py", B5B_OLD, B5B_NEW),
    ("Bug5b2 告警时自增", "snowflake_B.py", B5B2_OLD, B5B2_NEW),
    ("Bug4 早期加密采样", "train_lock60.py", B4_OLD, B4_NEW),
]


def check():
    print("=" * 78)
    print("  补丁检查（不改文件）")
    print("=" * 78)
    ok = True
    for name, fn, old, new in PATCHES:
        if not os.path.isfile(fn):
            print(f"  [跳过] {name:44} 文件不存在: {fn}")
            continue
        s = io.open(fn, encoding="utf-8").read()
        if new in s:
            print(f"  [已应用] {name}")
        elif old in s:
            print(f"  [待应用] {name}")
            ok = False
        else:
            print(f"  [?] {name:44} 未匹配到旧代码（可能已改过）")
    print()
    if ok:
        print("  所有补丁已就位")
    else:
        print("  存在待应用补丁，运行 --apply")
    return ok


def apply():
    print("=" * 78)
    print("  应用补丁")
    print("=" * 78)
    for name, fn, old, new in PATCHES:
        if not os.path.isfile(fn):
            print(f"  [跳过] {name}: {fn} 不存在")
            continue
        s = io.open(fn, encoding="utf-8").read()
        if new in s:
            print(f"  [已应用] {name}")
            continue
        if old not in s:
            print(f"  [失败] {name}: 未匹配旧代码")
            continue
        bak = fn + ".bak_patchall"
        if not os.path.exists(bak):
            shutil.copy2(fn, bak)
        io.open(fn, "w", encoding="utf-8").write(s.replace(old, new))
        print(f"  [OK] {name}  (备份 {bak})")
    print()
    print("  补丁应用完成。下一步: python patch_all.py --verify")


def verify():
    print("=" * 78)
    print("  验证补丁")
    print("=" * 78)

    # 1. 语法
    import py_compile
    for fn in TARGETS:
        if os.path.isfile(fn):
            try:
                py_compile.compile(fn, doraise=True)
                print(f"  [OK] 语法: {fn}")
            except Exception as e:
                print(f"  [ERR] 语法: {fn} -> {e}")
                return False

    # 2. key_map 含核心量
    s = io.open("run_metrics.py", encoding="utf-8").read()
    for k in ["eff_organs", "wiring_var_norm"]:
        print(f"  [{'OK' if k in s else 'ERR'}] key_map 含 {k}")

    # 3. H2 用 norm
    print(f"  [{'OK' if 'wiring_var_norm' in s and 'H2_跨样本方差' in s else 'ERR'}] "
          f"H2 已改用 wiring_var_norm")

    # 4. 纯逻辑验证：归一化是否让阈值可达
    print()
    print("─" * 78)
    print("  归一化前后对比（N=384，实测 wvar=0.00007）")
    print("─" * 78)
    n = 384
    _m = 1.0 / n
    vmax = (1.0 / n) * (1 - _m) ** 2 + ((n - 1) / n) * _m ** 2
    raw = 0.00007
    print(f"  var_max            = {vmax:.6f}")
    print(f"  实测 wiring_var    = {raw:.6f}")
    print(f"  归一化后 var_norm  = {raw / vmax:.4f}")
    print()
    print(f"  旧判据 raw > 0.05  → {'pass' if raw > 0.05 else 'fail'}  "
          f"（需 {0.05 / raw:.0f}x，不可达）")
    print(f"  新判据 norm > 0.05 → {'pass' if raw / vmax > 0.05 else 'fail'}  "
          f"（量纲可达，结果真实）")
    return True


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--check"
    {"--check": check, "--apply": apply, "--verify": verify}.get(
        cmd, lambda: print(__doc__))()
