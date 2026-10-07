"""
Dense baseline vs Snowflake MoE —— 配对统计检验
==============================================
README 的 claim:
    "achieving strong generalization under a fraction of the
     parameters of a dense baseline"

本脚本做三件事：
  1. 参数量对比（是不是真的 "a fraction of"）
  2. Loss/PPL 配对 t 检验（同 seed 配对，消除 seed 噪声）
  3. 计算参数效率比（每单位参数带来的 loss 降低）

用法:
    python compare_dense.py --dense output/dense --snow 1.788
"""
import argparse
import json
import os

import numpy as np


def paired_t(a, b):
    """配对 t 检验。返回 (diff_mean, t, df, dz, ci)"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    m = d.mean()
    s = d.std(ddof=1)
    se = s / np.sqrt(n) if s > 0 else 0.0
    t = m / se if se > 0 else float("inf")
    dz = m / s if s > 0 else float("inf")
    # df=n-1 的 95% 临界值（近似，n=10 时为 2.262）
    tc = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
          7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228}.get(n, 2.0)
    ci = (m - tc * se, m + tc * se)
    return m, t, n - 1, dz, ci, s


def load_dense(outdir):
    """读 dense_baseline.py 落盘的 10 个 txt"""
    rows = []
    for s in range(10):
        p = os.path.join(outdir, f"dense_seed{s}.txt")
        if not os.path.exists(p):
            continue
        d = {}
        for line in open(p):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                d[k] = v
        rows.append({"seed": int(d["seed"]),
                     "params": int(d["params"]),
                     "val_loss": float(d["val_loss"]),
                     "ppl": float(d["ppl"])})
    return sorted(rows, key=lambda r: r["seed"])


# Snowflake MoE 10 seeds 实测（来自 RESULTS.md）
SNOW_LOSS = {0: 1.8068, 1: 1.7716, 2: 1.7972, 3: 1.7841, 4: 1.7985,
             5: 1.7880, 6: 1.7784, 7: 1.8049, 8: 1.7671, 9: 1.7854}
SNOW_PARAMS = 5_163_618      # ★ Snowflake tiny 实测（L=4,n=384,d=128,h=32,r=4,cells=4）
                             # 注意：不是仓库 README 的 1,823,077（那是 v1.0 CellMoE 词级）
SNOW_EFF = {0: 25.4, 1: 25.9, 2: 23.0, 3: 28.7, 4: 26.7,
            5: 24.4, 6: 24.2, 7: 28.0, 8: 27.9, 9: 23.1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", type=str, default="output/dense")
    ap.add_argument("--snow", type=float, default=None,
                    help="Snowflake 单值 loss（无 10 seeds 时用）")
    ap.add_argument("--snow-params", type=int, default=SNOW_PARAMS)
    a = ap.parse_args()

    dense = load_dense(a.dense)
    print("=" * 88)
    print("  Dense Baseline vs Snowflake MoE")
    print("=" * 88)

    if not dense:
        print(f"  ❌ 未找到 dense 结果：{a.dense}")
        print("     先跑: python dense_baseline.py --data <txt> --sweep")
        return

    dp = dense[0]["params"]
    sp = a.snow_params
    print(f"\n【1】参数量")
    print(f"  Dense     : {dp:,}")
    print(f"  Snowflake : {sp:,}")
    ratio = sp / dp
    print(f"  比值      : {ratio:.4f}  ({'Snowflake 更少 ✅' if ratio < 1 else 'Snowflake 更多 ❌'})")
    print(f"  ⇒ README 声称 'a fraction of the parameters of a dense baseline'")
    print(f"    {'成立 ✅' if ratio < 1 else '不成立 ❌ —— 必须修改 README'}")

    print(f"\n【2】Loss / PPL")
    dl = [r["val_loss"] for r in dense]
    dpl = [r["ppl"] for r in dense]
    print(f"  Dense     : loss={np.mean(dl):.5f} ± {np.std(dl, ddof=1):.5f}   "
          f"ppl={np.mean(dpl):.4f}")

    if len(dense) >= 2:
        sl = [SNOW_LOSS[r["seed"]] for r in dense if r["seed"] in SNOW_LOSS]
        if len(sl) == len(dl):
            m, t, df, dz, ci, sd = paired_t(dl, sl)
            print(f"  Snowflake : loss={np.mean(sl):.5f} ± {np.std(sl, ddof=1):.5f}")
            print(f"\n  配对差 (dense − snow) = {m:+.5f}  SD={sd:.5f}")
            print(f"  t = {t:.2f}  df = {df}  Cohen's d_z = {dz:.2f}")
            print(f"  95% CI = [{ci[0]:+.5f}, {ci[1]:+.5f}]")
            if m > 0 and ci[0] > 0:
                print(f"\n  ⇒ Snowflake loss【显著更低】✅  (CI 不跨 0)")
                print(f"     ppl 差距: {np.mean(dpl):.3f} → "
                      f"{np.exp(np.mean(sl)):.3f}  "
                      f"(降低 {(1-np.exp(np.mean(sl))/np.mean(dpl))*100:.1f}%)")
            elif m < 0 and ci[1] < 0:
                print(f"\n  ⇒ Snowflake loss【显著更高】❌  README 的 claim 需修改")
            else:
                print(f"\n  ⇒ 无显著差异 ⚠")
        else:
            print(f"  (Snowflake seeds 不匹配，跳过配对检验)")

    print(f"\n【3】参数效率")
    print(f"  Dense     : {np.mean(dl):.5f} loss / {dp:,} params")
    print(f"  Snowflake : {np.mean(sl) if len(dense) >= 2 else 0:.5f} loss / {sp:,} params")
    if len(dense) >= 2:
        eff_d = np.mean(dl) / dp
        eff_s = np.mean(sl) / sp
        print(f"  每 M 参数的 loss: dense={eff_d*1e6:.4f}  snow={eff_s*1e6:.4f}")
        print(f"  ⇒ Snowflake 参数效率 {'更高 ✅' if eff_s < eff_d else '更低'}")

    print(f"\n【4】README claim 的最终判定")
    ok_param = ratio < 1
    ok_loss = (len(dense) >= 2 and 'm' in dir() and m > 0 and ci[0] > 0) or (
        a.snow is not None and np.mean(dl) > a.snow)
    print(f"  'fraction of the parameters' : {'✅' if ok_param else '❌'}")
    print(f"  'strong generalization'      : {'✅' if ok_loss else '❌'}")
    if ok_param and ok_loss:
        print(f"\n  ⇒ claim 可保留 ✅")
    else:
        print(f"\n  ⇒ claim 需修改 ⚠  建议改为:")
        print(f"     'achieves comparable perplexity with {ratio:.2f}x "
              f"the parameters of a dense baseline'")


if __name__ == "__main__":
    main()
