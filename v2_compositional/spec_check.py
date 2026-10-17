# -*- coding: utf-8 -*-
"""
spec_check.py —— 按《核心思想传输》第十一条规格逐条数值校验

只做三件事：
  A. 用户的初始化 base + randn*0.5 是否真的让 cos 落在 band=0.5
     （并认领：我 lock47 用的是 randn*0.5 无共享基，是错的）
  B. 高斯带通 vs 帐篷+铰链：各处 cos 上到底有没有"回复力"
  C. 稳态归一化 c/(S+td)*td 是单边还是双边；td 要不要随 N 缩放
"""
import numpy as np

rng = np.random.default_rng(0)
BAND, WIDTH = 0.5, 0.1


def gram(M):
    N = M / np.linalg.norm(M, axis=-1, keepdims=True)
    return N @ N.T


# ------------------------------------------------------------------ A
print("=" * 92)
print("【A】初始化：用户规格 base(0.5√d) + randn*0.5  vs  我 lock47 的 randn*0.5")
print("=" * 92)


def init_spec(n, d, r):
    """用户规格：base = normalize(randn(d)) * 0.5 * sqrt(d)；+ randn(n,d)*0.5"""
    b = r.normal(size=d)
    b = b / np.linalg.norm(b) * 0.5 * np.sqrt(d)
    return b + r.normal(size=(n, d)) * 0.5


def init_mine(n, d, r):
    """我 lock47 用的：randn(n,d) * 0.5 —— 无共享基"""
    return r.normal(size=(n, d)) * 0.5


print(f"{'N':>5}{'d':>6}{'初始化':>16}{'cos均值':>10}{'cos标准差':>11}"
      f"{'带内占比':>10}")
print("-" * 92)
for N in [8, 384, 512]:
    for d, nm, fn in [(128, "用户规格", init_spec), (128, "我lock47", init_mine)]:
        M = fn(N, d, rng)
        G = gram(M)
        off = ~np.eye(N, dtype=bool)
        inb = ((G[off] > BAND - 2 * WIDTH) & (G[off] < BAND + 2 * WIDTH)).mean()
        print(f"{N:>5}{d:>6}{nm:>16}{G[off].mean():>10.4f}{G[off].std():>11.4f}"
              f"{inb:>10.4f}")
    print("-" * 92)
print("→ 用户规格 cos≈0.50 与 N 无关（N=512 也成立）。我 lock47 的 cos≈0，in_band=0。")
print("  ⚠ 这是我偏离规格的地方，已在 ablate_permute.py 改用共享基。确认归属：你对。")

# ------------------------------------------------------------------ B
print("\n" + "=" * 92)
print("【B】回复力：三种连接函数在 cos 轴上哪里有力（这是关键分歧点）")
print("=" * 92)
print("""
  u = (cos - band)/width
  高斯带通 : c = exp(-u²)              dc/dcos = -2u/w · exp(-u²)
  帐篷     : c = relu(2-|u|)           dc/dcos = ∓1/w  (|u|<2), 0 (|u|>2)
  铰链自愈 : heal = relu(|u|-2)²·w²/N² d(heal)/dcos = 2·sign(u)(|u|-2)·w/N²
""")
print(f"{'cos':>6}{'u':>7} | {'高斯 c':>11}{'|dc/dcos|':>12} | {'帐篷 c':>9}{'|dc/dcos|':>11} "
      f"| {'铰链 |dheal/dcos|':>18}")
print("-" * 92)
N_REF = 384
for cos in [-0.1, 0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
    u = (cos - BAND) / WIDTH
    cg = np.exp(-u ** 2)
    dg = abs(-2 * u / WIDTH * cg)
    ct = max(0.0, 2 - abs(u))
    dt = (1 / WIDTH) if abs(u) < 2 else 0.0
    dh = abs(2 * np.sign(u) * max(0.0, abs(u) - 2) * WIDTH / N_REF ** 2) \
        if abs(u) > 2 else 0.0
    print(f"{cos:>6.2f}{u:>7.1f} | {cg:>11.2e}{dg:>12.2e} | {ct:>9.3f}{dt:>11.1f} "
          f"| {dh:>18.2e}")

print("\n★ 关键读数：")
print(f"  cos=0.9（趋同坍缩区）: 高斯 |dc/dcos| = "
      f"{abs(-2*4/WIDTH*np.exp(-16)):.2e}   帐篷 = 0   铰链 = "
      f"{2*2*WIDTH/N_REF**2:.2e}")
print(f"  cos=0.0（器官孤立区）: 高斯 |dc/dcos| = "
      f"{abs(-2*(-5)/WIDTH*np.exp(-25)):.2e}   帐篷 = 0   铰链 = "
      f"{2*3*WIDTH/N_REF**2:.2e}")
print("""
→ 红线3「帐篷有死区、梯度会断」—— 数值上成立，帐篷在 |u|>2 处梯度严格为 0。
→ 但高斯在远离带处梯度是【数值≈0】而非【严格为 0】：
     cos=0.9 -> 9e-7     cos=0.0 -> 1.4e-9
   数学上非零，实际等于没有力。
→ 结论：高斯解决了「硬死区」，但没有解決「没有回复力」。
   原 tent+hinge 的铰链在 |u|>2 处线性增长永不衰减，那才是"拉得回来"的东西。
   换成高斯后，趋同坍缩（病理3）失去了【显式】的回复力，只剩任务损失的隐式压力。
""")

# ------------------------------------------------------------------ C
print("=" * 92)
print("【C】稳态归一化 c/(S+td)*td：单边还是双边？td 要不要随 N 缩放？")
print("=" * 92)


def steady(c_raw, td):
    S = c_raw.sum(-1, keepdims=True)
    return c_raw / (S + td) * td


N = 384
print(f"设所有器官都在带内（c_raw=1），N={N}")
print(f"{'td':>8}{'行和(输出)':>14}{'connect_mean':>14}{'黄金区间0.3~0.7?':>18}")
print("-" * 92)
for td in [0.5, 1, 4, 8, 32, 128, 383, 768]:
    c1 = np.ones((N, N)) - np.eye(N)          # 对角 cos=1 -> 高斯≈0
    o = steady(c1, td)
    cm = o.mean()
    ok = "✅" if 0.3 <= cm <= 0.7 else "❌"
    print(f"{td:>8}{o.sum(-1).mean():>14.3f}{cm:>14.4f}{ok:>18}")
print(f"\n→ 行和上限 = td（S→∞ 时行和→td）。所以 td 是【每个器官的连接总配额】，")
print(f"  语义清晰、与 N 无关。但 connect_mean ≈ td/(N-1+td)：")
print(f"  td=4, N=384 -> connect_mean={4/(383+4):.4f}，远低于黄金区间 0.3~0.7。")
print(f"  要 connect_mean=0.5 需 td≈N-1={N-1}。")
print("\n→ 两种解释，结论相反，需要你拍板：")
print("   解释1: td = 每个器官的连接总配额（常数，如 4） -> connect_mean≈0.01")
print("   解释2: td ~ N-1（随规模缩放）                 -> connect_mean≈0.5")
print("   ⚠ 若按解释1，N=384 时每个器官平均只分到 4/384=0.01 的连接权重，")
print("     连接机制几乎不起作用（c.T@mem ≈ 0.01·mem，被 mem 淹没）。")

print("\n  单边性验证（S 是否能被抬起来）：")
for S0, tag in [(1e-6, "几乎无连接"), (10.0, "稀疏"), (383.0, "全连接")]:
    c_raw = np.zeros((1, N)); c_raw[0, :int(S0)] = 1.0 if S0 >= 1 else S0
    o = steady(c_raw, td=4.0)
    print(f"    S_in={S0:>10.4g} ({tag:<10}) -> S_out={o.sum():.4f}  "
          f"{'被抬起' if o.sum() > S0 * 1.5 else '未被抬起（单边封顶）'}")
print("\n→ 归一化只【封顶】不【托底】：连接少了它不管，连接多了它压回来。")
print("  这与『稳态区间』的下界（不能太低/器官孤立）不矛盾——下界靠初始化保证。")

# ------------------------------------------------------------------ D
print("\n" + "=" * 92)
print("【D】折叠恒等复核（规格第三节）：raw_sim + raw_sim@c ≡ x@(mem + c.T@mem).T")
print("=" * 92)
K, D, B = 16, 32, 48
mem = init_spec(K, D, rng)
G = gram(mem)
u = (G - BAND) / WIDTH
c = np.exp(-u ** 2)
c = steady(c, td=4.0)
x = rng.normal(size=(B, D))
raw = x @ mem.T / np.sqrt(D)
A = raw + raw @ c
Bv = x @ (mem + c.T @ mem).T / np.sqrt(D)
Bw = x @ (mem + c @ mem).T / np.sqrt(D)
print(f"max|c - c.T|            = {np.abs(c - c.T).max():.4f}   ← 行归一化破坏对称性")
print(f"用 c.T  折叠 logits 差  = {np.abs(A - Bv).max():.2e}   ✅ 规格正确")
print(f"用 c    折叠 logits 差  = {np.abs(A - Bw).max():.2e}   ← 会换成另一个机制")
print("\n→ 规格里写的是 c.T @ self.organelle_memory ✅，与我上轮抓到的 bug 一致。")

# ------------------------------------------------------------------ E
print("\n" + "=" * 92)
print("【E】高斯带通的一个白送性质：自连接自动归零")
print("=" * 92)
print(f"对角 cos=1 -> u={(1-BAND)/WIDTH:.1f} -> c_diag=exp(-{((1-BAND)/WIDTH)**2:.0f})="
      f"{np.exp(-((1-BAND)/WIDTH)**2):.2e}")
print("→ 高斯下对角天然≈0，不需要 eye mask（帐篷同理 u=5 也归零）。")
print("  这条比 tent 更干净：无需额外 mask 代码。")
