# -*- coding: utf-8 -*-
"""给 train_v3.py 打补丁：训练结束后直接测量 wiring 的边际分布。
不靠推算，直接量：
  - eff_per_sample  每个样本激活多少器官
  - eff_marginal    跨样本一共用了多少器官（池子大小）
  - identity_gap    恒等式检验，应≈0，否则指标本身有 bug
"""
import io, sys

P = "train_v3.py"
s = io.open(P, encoding="utf-8").read()

if "wiring_probe" in s:
    print("[SKIP] 已打过补丁"); sys.exit(0)

ANCHOR = "    val_loss = evaluate(model, get_batch, 20, device, bf16)"
if ANCHOR not in s:
    print("[ERR] 找不到锚点"); sys.exit(1)

PROBE = ANCHOR + '''

    # ---- wiring 探针：直接测量，不靠推算 ----
    probe = {}
    try:
        model.eval()
        acc_sum = None; acc_sq = None; acc_n = 0
        with torch.no_grad():
            for _ in range(12):
                xb, _ = get_batch("val")
                xb = xb.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    _ = model(xb)
                for mod in model.modules():
                    lm = getattr(mod, "_last_mix", None)
                    if lm is None:
                        continue
                    w = lm.detach().float()
                    if w.dim() != 2:
                        continue
                    sm = w.sum(0); sq = (w * w).sum(0)
                    acc_sum = sm if acc_sum is None else acc_sum + sm
                    acc_sq = sq if acc_sq is None else acc_sq + sq
                    acc_n += w.shape[0]
        if acc_sum is not None and acc_n > 0:
            N = int(acc_sum.numel())
            m = acc_sum / acc_n
            E2 = acc_sq / acc_n
            var_i = (E2 - m * m).clamp(min=0)
            sum_var = float(var_i.sum())
            sum_m2 = float((m * m).sum())
            E_sumw2 = float(E2.sum())
            probe = {
                "N_organs": N,
                "n_samples": int(acc_n),
                "sum_var": sum_var,
                "sum_m2": sum_m2,
                "E_sumw2": E_sumw2,
                "identity_gap": (sum_var + sum_m2) - E_sumw2,
                "eff_per_sample": 1.0 / E_sumw2,
                "eff_marginal": 1.0 / sum_m2 if sum_m2 > 0 else None,
                "pool_ratio": (1.0 / sum_m2) / (1.0 / E_sumw2) if sum_m2 > 0 else None,
                "var_norm_probe": (sum_var / N) / ((1.0 / N) * (1 - 1.0 / N)),
            }
        model.train()
    except Exception as e:
        probe = {"error": repr(e)}'''

s = s.replace(ANCHOR, PROBE)
s = s.replace('        "leak_ok": not a.no_leak_check,',
              '        "leak_ok": not a.no_leak_check,\n        "wiring_probe": probe,')

# 打印探针结果
s = s.replace('    print("=" * 70)\n\n    os.makedirs(a.outdir',
              '    if probe and "error" not in probe:\n'
              '        print(f"    [探针] 每样本激活 = {probe[\\'eff_per_sample\\']:.2f}   "\n'
              '              f"跨样本池 = {probe[\\'eff_marginal\\']:.1f}   "\n'
              '              f"池/样本 = {probe[\\'pool_ratio\\']:.2f}x")\n'
              '        print(f"    [探针] 恒等式残差 = {probe[\\'identity_gap\\']:.2e} (应≈0)")\n'
              '    print("=" * 70)\n\n    os.makedirs(a.outdir')

io.open(P, "w", encoding="utf-8").write(s)
import subprocess
r = subprocess.run([sys.executable, "-c", f"import ast;ast.parse(open('{P}',encoding='utf-8').read())"],
                   capture_output=True, text=True)
print("[OK] 补丁完成，语法通过" if r.returncode == 0 else "[ERR] 语法错误\n" + r.stderr)
