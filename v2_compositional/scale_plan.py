# -*- coding: utf-8 -*-
"""
scale_plan.py —— N=384/512、P=32/128 的真实成本账本

回答一个问题：用户拍的配置（N=384, P=32）到底比 Top-K 便宜还是贵？
答案取决于一件之前没说死的事：delta 是不是低秩。
"""
import numpy as np

D, H, K_TOPK = 128, 32, 4          # d, h, Top-K 的 k
SEQ_LEN = 128
DH = D * H


def pz_per_token(N, P, r=None, d=D, h=H):
    """拼专家每 token 的 FLOPs（乘加数）。

    r=None → delta 是满秩 [d,h]：合成 einsum('ci,idh->cdh') 成本 N·d·h
    r=int  → delta = U_i @ V，U_i:[d,r] V:[r,h]
             合成 = Σmix_i U_i (N·d·r) + U_wired@V (d·r·h)
    """
    if r is None:
        synth = N * d * h                      # 每 chunk 一次
    else:
        synth = N * d * r + d * r * h
    return synth / P + d * h                   # 应用：每 token 一次 d·h


def topk_per_token(k=K_TOPK, d=D, h=H):
    return k * d * h


BASE = topk_per_token()
print(f"基准：Top-K k={K_TOPK} 每 token = {BASE:,} FLOPs\n")

print("=" * 92)
print("【核心结论】N=384 在 P=32 下，低秩与否决定胜负")
print("=" * 92)
print(f"{'配置':<34}{'每token FLOPs':>16}{'相对 Top-K':>14}{'判定':>14}")
print("-" * 92)
rows = [
    ("N=384, P=32,   delta 满秩",   384, 32, None),
    ("N=384, P=32,   delta 低秩 r=4", 384, 32, 4),
    ("N=384, P=128,  delta 满秩",   384, 128, None),
    ("N=384, P=128,  delta 低秩 r=4", 384, 128, 4),
    ("N=512, P=128,  delta 低秩 r=4", 512, 128, 4),
    ("N=512, P=128,  delta 低秩 r=8", 512, 128, 8),
    ("N=8,   P=32,   delta 低秩 r=4", 8,   32,  4),
]
for nm, N, P, r in rows:
    f = pz_per_token(N, P, r)
    ratio = f / BASE
    j = "碾压" if ratio < 0.7 else ("便宜" if ratio < 1.0 else "更贵 ⚠")
    print(f"{nm:<34}{f:>16,.0f}{ratio:>13.3f}x{j:>14}")

print("\n→ 满秩 delta 下 N=384+P=32 是 3.25x 更贵（越开大 N 越亏）。")
print("  低秩 r=4 把它翻成 0.66x 便宜。**低秩不是优化项，是这个配置成立的前提。**")

# ---------------------------------------------------------------- P 扫描
print("\n" + "=" * 92)
print("【P 扫描】N=384, r=4：P 该取多少")
print("=" * 92)
print(f"{'P':>6}{'每token':>14}{'相对':>10}{'chunks/seq':>12}  判定")
for P in [1, 8, 16, 32, 64, 128]:
    f = pz_per_token(384, P, 4)
    nch = SEQ_LEN // P
    j = "更贵" if f / BASE > 1 else ("临界" if f / BASE > 0.7 else "便宜")
    print(f"{P:>6}{f:>14,.0f}{f/BASE:>9.3f}x{nch:>12}  {j}")
print(f"\n→ 盈亏平衡 P* = N·d·r/(k·d·h - d·h) = "
      f"{384*D*4/((K_TOPK-1)*DH):.1f}（再考虑 d·r·h 项会略大）")
print("  P=32 已经能赢；P=128(整条序列一份 mix) 赢最多，但 mix 分辨率最粗。")

# ---------------------------------------------------------------- 参数量
print("\n" + "=" * 92)
print("【参数量】瓶颈从 FLOPs 转移到参数量 —— 这正是命题的核心，但必须报出来")
print("=" * 92)
def pz_params(N, r, d=D, h=H, n_cells=4, L=4):
    per_cell = (d * h                      # W1_base
                + N * d * r + r * h        # W1 U,V
                + h * d                    # W2_base
                + N * h * r + r * d        # W2 U,V
                + N * d)                   # organelle_memory (tent/hinge 作用对象)
    return per_cell * n_cells * L

def topk_params(N, d=D, h=H, n_cells=4, L=4):
    per_cell = 2 * N * d * h               # N 个满秩专家 W1/W2
    return per_cell * n_cells * L

print(f"{'配置':<34}{'参数量':>16}{'相对 Top-K(8专家)':>20}")
tk8 = topk_params(8)
for N, r in [(8, 4), (384, 4), (512, 4)]:
    p = pz_params(N, r)
    print(f"{f'拼专家 N={N}, r={r}':<34}{p:>16,}{p/tk8:>19.2f}x")
print(f"{'Top-K N=8 (满秩专家)':<34}{tk8:>16,}{1.0:>19.2f}x")
print(f"{'Top-K N=384 (等表达，理论)':<34}{topk_params(384):>16,}"
      f"{topk_params(384)/tk8:>19.2f}x")

print("\n→ 拼专家 N=384 的参数只有 Top-K N=384 的 "
      f"{pz_params(384,4)/topk_params(384):.3f}，因为低秩。")
print("  但 FLOPs 只有 Top-K k=4 的 0.66x。这就是「瓶颈从 FLOPs 转移到参数量」。")
print("  ⚠ 公平对比必须同时给：等参数基线 + 等 FLOPs 基线，两套都要。")

# ---------------------------------------------------------------- 显存
print("\n" + "=" * 92)
print("【显存 / 激活】N=384 能不能装下")
print("=" * 92)
B = 64
for N, P in [(384, 32), (384, 128), (512, 128)]:
    C = B * (SEQ_LEN // P)
    # 合成后的权重 [C,d,h] + 输入 [C,P,d] + 输出
    act = C * D * H * 2 + C * P * D + C * P * H
    param_mb = pz_params(N, 4) * 4 / 1024**2
    print(f"N={N:>4} P={P:>4}  chunks={C:>5}  "
          f"合成权重激活={C*D*H*4/1024**2:>7.1f}MB  参数={param_mb:>7.1f}MB")
print("\n→ P=128 时 chunks=B=64，合成权重 [64,128,32] 只有 1MB，完全没问题。")
print("  P=32 时 chunks=256，4MB。都不构成瓶颈。")

# ---------------------------------------------------------------- 三能力
print("\n" + "=" * 92)
print("【三种能力】N=384 如何天然支撑（架构层保证，不是调参）")
print("=" * 92)
print("""
1. 终身学习 —— 新增器官，不动旧的
   N: 384 -> 384+M。旧器官 U/V 冻结，只训新器官 + mix 输出层。
   拼专家独有优势：mix 是连续分布，新器官可以被【渐进纳入】，
   不存在 Top-K 的"抢占"（新专家抢走 token 导致旧专家饿死 => 灾难性遗忘）。
   判据：旧域回归 <= 0.5，新域下降 >= 20%

2. 可擦除 —— 置零 mix 的第 j 维（核态限基）
   器官 j 的唯一入口就是 mix_j。置零 = 该器官永久不参与任何组合。
   比擦 memory_value 干净：memory 不是计算通路的一部分，擦了模型还能跑。
   判据（待你定方向）：擦除目标域 PPL 显著上升，且非目标域 PPL 基本不变

3. 组合泛化 —— 新输入 = 新的 mix 组合
   N-1 维凸包。N=384 -> 383 维组合空间（N=8 只有 7 维）。
   判据：mix_gain(learned) > mix_gain(permuted)，CI 下界 > 0
""")

# ---------------------------------------------------------------- warmup
print("=" * 92)
print("【warmup】3000 步够不够（N=384 的 fitness 收敛）")
print("=" * 92)
print(f"""
你定了 3000 步。粗估：N=384 时每个器官平均只被 {1/384:.4f} 的权重分到，
fitness_ema 的 EMA 时间常数若为 0.99，达到 95% 收敛需 ln(0.05)/ln(0.99) ≈ 300 次有效观测。
batch=64, seq=128, P=32 -> 每步 {64*(128//32):,} 个 chunk。
3000 步 -> {3000*64*(128//32):,} 个 chunk，每器官平均 {3000*64*(128//32)/384:,.0f} 次观测。
=> 3000 步是充分的（远超 300 次阈值）。但 fitness 的【分化】需要更久，
   建议剪枝前先看 fitness_ema 的基尼系数是否稳定，而不是只看步数。
""")
