# -*- coding: utf-8 -*-
"""
permute_check.py —— 用户给的 permuted 操作定义能不能测出东西

用户原话：「permuted 组：训练时随机打乱器官顺序（每次 forward 前）」

验证三种打法，看哪种真的改变输出：
  A. 一致 permute (organelle_memory, W1_all, W2_all)  ← 用户字面描述
  B. 只 permute W1/W2，不 permute organelle_memory
  C. permute mix 的行（chunk → 配方的配对关系）

判据：输出若完全不变，则该对照测不出任何东西（数学恒等）。
"""
import numpy as np

rng = np.random.default_rng(0)
D, H, K, B = 128, 32, 16, 64
BAND, W, RHO, TEMP = 0.5, 0.1, 0.3, 2.0


def gram(M):
    N = M / np.linalg.norm(M, axis=-1, keepdims=True)
    return N @ N.T


def tent(mem):
    u = (gram(mem) - BAND) / W
    c = np.maximum(0.0, 2.0 - np.abs(u))
    c = c / (c.sum(-1, keepdims=True) + RHO) * RHO
    return u, c


def sm(M, axis=-1):
    e = np.exp(M - M.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def forward(x, mem, W1, W2, perm=None, perm_mix=None, temp=TEMP):
    """x:[B,d] mem:[K,d] W1:[K,d,h] W2:[K,h,d]
    perm     : 对器官维度的一致重排（同时作用于 mem 和 W）
    perm_mix : 对 mix 行的重排（chunk→配方配对）
    """
    if perm is not None:
        mem = mem[perm]
        W1 = W1[perm]
        W2 = W2[perm]
    u, c = tent(mem)
    key = mem + c.T @ mem
    mix = sm(x @ key.T / np.sqrt(D) * temp)          # [B, K]
    if perm_mix is not None:
        mix = mix[perm_mix]
    W1w = np.einsum("bi,idh->bdh", mix, W1)
    h = np.einsum("bd,bdh->bh", x, W1w)
    h = np.tanh(h)
    W2w = np.einsum("bi,ihd->bhd", mix, W2)
    return np.einsum("bh,bhd->bd", h, W2w), mix


x = rng.normal(size=(B, D))
mem = rng.normal(size=(K, D)) * 0.5
W1 = rng.normal(size=(K, D, H)) * 0.1
W2 = rng.normal(size=(K, H, D)) * 0.1

base, mix_base = forward(x, mem, W1, W2)

print("=" * 92)
print("【A】一致 permute (mem, W1, W2) —— 用户字面描述")
print("=" * 92)
for trial in range(3):
    p = rng.permutation(K)
    o, m = forward(x, mem, W1, W2, perm=p)
    print(f"trial{trial}  max|out - base| = {np.abs(o - base).max():.3e}   "
          f"max|mix - mix_base| = {np.abs(m - mix_base).max():.3e}")
print("\n→ 输出完全不变。因为 softmax 与 einsum 对器官维都是【置换等变】的：")
print("    mem[π] → key[π] → mix[π] → Σ mix[π]_i W[π]_i = Σ mix_i W_i")
print("  ⚠ 按字面实现 permuted 组，loss 曲线会与主组【逐位相同】，白跑一趟。")

print("\n" + "=" * 92)
print("【B】只 permute W，不动 mem（破坏身份-权重绑定）")
print("=" * 92)
for trial in range(3):
    p = rng.permutation(K)
    o, m = forward(x, mem, W1[p], W2[p])
    print(f"trial{trial}  max|out - base| = {np.abs(o - base).max():.3e}   "
          f"mean|out-base| = {np.abs(o - base).mean():.3e}")
print("\n→ 改变了输出，但它测的是'器官身份与权重的绑定'，不是'分工'。")
print("  而且训练时每次 forward 换一个 π ⇒ 梯度方向每步跳变 ⇒ 训不出来。")
print("  这样比出来的差距只能归因于'训练被破坏'，结论不可引用。")

print("\n" + "=" * 92)
print("【C】permute mix 的行（chunk→配方配对）—— 这才是正确的对照")
print("=" * 92)
for trial in range(3):
    pm = rng.permutation(B)
    o, m = forward(x, mem, W1, W2, perm_mix=pm)
    print(f"trial{trial}  max|out - base| = {np.abs(o - base).max():.3e}   "
          f"mean|out-base| = {np.abs(o - base).mean():.3e}   "
          f"mix 边缘分布差 = {np.abs(np.sort(m,0)-np.sort(mix_base,0)).max():.3e}")
print("\n→ 输出改变，且 mix 的【边缘分布完全不变】（排序后差为 0）。")
print("  这正是要的：只破坏'内容→配方'的对应，不破坏任何其他统计量。")

# ------------------------------------------------------------------ 定量
print("\n" + "=" * 92)
print("【定量】在真实任务上，三种对照的 loss 差")
print("=" * 92)
Ad = rng.normal(size=(D, D)) / np.sqrt(D)
Cd = rng.normal(size=(D, D)) / np.sqrt(D)


def task(xx):
    return np.tanh(xx @ Ad) @ Cd


# 造一个"mix 真的有用"的 ground truth：让 W1 沿 x 的某个方向分化
dirs = rng.normal(size=(K, D))
dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
W1s = W1 + 0.3 * dirs[:, :, None] * rng.normal(size=(1, H))
W2s = W2

xtr = rng.normal(size=(1024, D))
ytr = task(xtr)


def mse(perm_mix=None, perm_organ=None):
    o, _ = forward(xtr, mem, W1s, W2s, perm=perm_organ, perm_mix=perm_mix)
    return ((o - ytr) ** 2).mean()


l_base = mse()
l_const = None
# 常数配方（退化对照）
o_const, _ = forward(xtr, mem, W1s, W2s)
mm = np.tile(mix_base.mean(0), (xtr.shape[0], 1))
W1w = np.einsum("bi,idh->bdh", mm, W1s)
hh = np.tanh(np.einsum("bd,bdh->bh", xtr, W1w))
W2w = np.einsum("bi,ihd->bhd", mm, W2s)
l_const = ((np.einsum("bh,bhd->bd", hh, W2w) - ytr) ** 2).mean()

l_perm = np.mean([mse(perm_mix=rng.permutation(xtr.shape[0])) for _ in range(20)])
l_permO = np.mean([mse(perm_organ=rng.permutation(K)) for _ in range(20)])

print(f"{'变体':<34}{'MSE':>16}{'相对 主组':>14}")
print("-" * 92)
print(f"{'主组（学出配方）':<34}{l_base:>16.6f}{0.0:>14.4f}")
print(f"{'permuted mix（破坏配对）':<34}{l_perm:>16.6f}{l_perm/l_base-1:>+13.2%}")
print(f"{'permuted 器官（字面实现）':<34}{l_permO:>16.6f}{l_permO/l_base-1:>+13.2%}")
print(f"{'常数配方（完全退化）':<34}{l_const:>16.6f}{l_const/l_base-1:>+13.2%}")
print("\n→ permuted 器官相对主组是 0.00% —— 再次确认它是恒等变换。")
print("  只有 permuted mix 才落在'主组'和'常数'之间，这才是有效的对照。")

print("\n" + "=" * 92)
print("【结论】")
print("=" * 92)
print("""
⚠ 用户给的判据操作定义要改：
   「训练时随机打乱器官顺序」⇒ 数学恒等，loss 逐位相同，测不出任何东西。

✅ 正确的 permuted 对照（两种，都要做，且都在【训练完成后】做）：

   对照1（推理时，最关键）：训练照常。评估时把 batch 内 mix 的行随机重排，
        使 chunk i 用 chunk j 的配方。mix 边缘分布不变，只破坏内容↔配方对应。
        判据：mix_gain(learned) > mix_gain(permuted)，5 seeds 配对 CI 下界 > 0

   对照2（训练时）：固定一个 π，全程用它。等价于给器官编号换个顺序，
        仍恒等。所以训练时做 permute 是无意义的 —— 这条应当取消。

   补充对照3（器官 shuffle 的正确形态）：把 organelle_memory 单独打乱、
        W 不动 ⇒ 测'身份-权重绑定'。但训练时会破坏梯度，只能做短程探针。
""")
