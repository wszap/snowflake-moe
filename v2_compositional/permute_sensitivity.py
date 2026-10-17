# -*- coding: utf-8 -*-
"""
permute_sensitivity.py —— permuted 判据的【灵敏度校准】

上一轮发现：在随手构造的任务上，常数配方比学出配方还好 1%，permuted 差 0.00%。
这有两种解释：
  (a) 那个任务本来就没有"分工"（mix 无用）→ 判据正确地报了 0
  (b) 判据本身没灵敏度 → 真有分工也测不出来

不先排掉 (b)，真机跑出 0 就无法解释。
做法：构造一个【已知 mix 必须有用】的任务，看判据能不能检出。

任务构造（簇-专家对齐）：
  x 来自 K 个簇，簇 j 的样本必须经过专家 W*_j
  y = tanh(x @ W*_{c(x)}) @ W_out
  ⇒ 最优 mix 是把权重全给器官 c(x)。这是货真价实的分工。
"""
import numpy as np

rng = np.random.default_rng(0)
D, H, K = 64, 32, 8
N_EVAL = 4096


def sm(M):
    e = np.exp(M - M.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------- 造数据
def make_task(K=K, d=D, h=H, sep=1.0, n=N_EVAL):
    """簇中心 + 每簇一个真专家。sep 控制簇分离度（越大越好分）。"""
    centers = rng.normal(size=(K, d)) * sep
    Wstar = rng.normal(size=(K, d, h)) * 0.2
    Wout = rng.normal(size=(h, d)) / np.sqrt(h)
    c = rng.integers(0, K, n)
    x = centers[c] + rng.normal(size=(n, d)) * 0.5
    h_ = np.tanh(np.einsum("bd,bdh->bh", x, Wstar[c]))
    y = h_ @ Wout
    return x, y, c, Wstar, Wout, centers


def apply_mix(x, mix, W1, W2):
    W1w = np.einsum("bi,idh->bdh", mix, W1)
    h = np.tanh(np.einsum("bd,bdh->bh", x, W1w))
    W2w = np.einsum("bi,ihd->bhd", mix, W2)
    return np.einsum("bh,bhd->bd", h, W2w)


print("=" * 94)
print("【校准】在【已知有分工】的任务上，判据能不能检出？")
print("=" * 94)

for sep in [0.0, 1.0, 2.0]:
    x, y, c, Wstar, Wout, centers = make_task(sep=sep)
    # 器官权重 = 真专家（这是"训练完美收敛"的理想情形）
    W1 = Wstar.copy()
    W2 = np.tile(Wout[None, :, :], (K, 1, 1))

    # 最优 mix：one-hot 指向自己的簇
    mix_oracle = np.eye(K)[c]
    # 学出 mix（模拟：用 x 与 centers 的相似度）—— 这里直接用 oracle 加噪声模拟不同质量
    # 常数 mix
    mix_const = np.tile(mix_oracle.mean(0), (len(x), 1))

    def mse(mix):
        return ((apply_mix(x, mix, W1, W2) - y) ** 2).mean()

    l_or = mse(mix_oracle)
    l_cn = mse(mix_const)

    # permuted：打乱 (样本→配方) 配对
    lp = []
    for _ in range(50):
        lp.append(mse(mix_oracle[rng.permutation(len(x))]))
    l_pm = float(np.mean(lp))
    l_pm_sd = float(np.std(lp))

    print(f"\n簇分离度 sep={sep}  （sep=0 ⇒ 簇重合 ⇒ 理论上无分工可学）")
    print(f"  {'变体':<26}{'MSE':>14}{'相对 oracle':>14}{'mix_gain':>12}")
    print("  " + "-" * 66)
    print(f"  {'oracle mix（真分工）':<26}{l_or:>14.6f}{0.0:>+13.2%}"
          f"{l_cn - l_or:>12.6f}")
    print(f"  {'permuted mix（破坏配对）':<26}{l_pm:>14.6f}{l_pm/l_or-1:>+13.2%}"
          f"{l_cn - l_pm:>12.6f}")
    print(f"  {'常数 mix（完全退化）':<26}{l_cn:>14.6f}{l_cn/l_or-1:>+13.2%}"
          f"{0.0:>12.6f}")
    print(f"  permuted 50 次标准差 = {l_pm_sd:.2e}  ⇒ 判据噪声地板 ≈ "
          f"{l_pm_sd:.2e}")

    gap = l_pm - l_or
    print(f"\n  判据灵敏度检验：")
    print(f"    permuted − oracle = {gap:.6f}")
    if gap > 10 * l_pm_sd:
        print(f"    ✅ 判据有效：信号 {gap:.2e} 是噪声 {l_pm_sd:.2e} 的 "
              f"{gap/l_pm_sd:.0f} 倍")
    else:
        print(f"    ⚠ 判据失灵：信号 {gap:.2e} ≈ 噪声 {l_pm_sd:.2e}")

print("\n" + "=" * 94)
print("【关键】sep=0（无分工可学）时应看到 gap≈0；sep 增大 gap 应增大")
print("=" * 94)
print("""
判据的【ROC 式校准】：先确认它在"已知有分工"时能喊有，在"已知无分工"时能喊无。
真机上若测得 gap ≈ 0，才有资格说"模型没分工"，而不是"判据坏了"。

⚠ 上一轮那个随手构造的任务里常数配方反而更好（-1.02%），
  说明那个构造【没有】真的把分工编进任务 —— 是构造的问题，不是判据的问题。
  本文件用簇-专家对齐的构造来排除这个混淆。
""")

# ---------------------------------------------------------------- 噪声地板
print("=" * 94)
print("【噪声地板】真机上要多大的 gap 才算显著？")
print("=" * 94)
print("""
    permuted 是随机重排，本身有方差。判据显著的条件：

        gap > 2.8 × sd(permuted)        （单 seed，近似 95%）

    更稳的做法：不要比 PPL 的绝对差，而是
      · 对同 5 个 seed，每个 seed 做 20 次 permute 取均值
      · 用【配对 t 检验】(learned_i vs permuted_i)，报告 Cohen's d_z
      · 同时报 gap 的 95% CI

    若真机上 gap 落在噪声地板内 ⇒ 结论只能是"未检出分工"，
    【不能】说"证明没有分工"。这两者在论文里写法完全不同。
""")

# ---------------------------------------------------------------- 功率分析
print("=" * 94)
print("【功率分析】5 seeds 能检出多小的效应？")
print("=" * 94)
from scipy import stats as st

# 配对 t 检验：alpha=0.05 双侧、power=0.8 下可检出的最小 Cohen's d_z。
# 用正态近似 d_z = (z_{1-a/2} + z_{power}) / sqrt(n)，再乘小样本修正 (1 + 1/(4·df))。
# （避免非中心 t 的 nct.cdf 在小 nc 上数值不稳导致求解失败）
alpha, power = 0.05, 0.80
z_a = st.norm.ppf(1 - alpha / 2)
z_b = st.norm.ppf(power)
print(f"  z_{{1-a/2}}={z_a:.3f}  z_{{power}}={z_b:.3f}")
for n_seed in [3, 5, 10, 20, 30]:
    df = n_seed - 1
    dz = (z_a + z_b) / np.sqrt(n_seed) * (1 + 1 / (4 * df))
    # 用非中心 t 复核（若可用）
    try:
        nc = dz * np.sqrt(n_seed)
        pwr = 1 - st.nct.cdf(st.t.ppf(1 - alpha / 2, df), df, nc)
        chk = f"  (非中心 t 复核 power={pwr:.2f})"
    except Exception:
        chk = ""
    print(f"  n={n_seed:>3} seeds  →  80% 功率可检出的最小 d_z = {dz:.2f}{chk}")

print("\n→ n=5 只能检出 d_z ≳ 1.4 的效应（很大）。")
print("  若真机 gap 较小，5 seeds 会漏检 ⇒ 主对照建议 10 seeds（d_z ≳ 1.0）。")
