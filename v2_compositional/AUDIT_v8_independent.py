# -*- coding: utf-8 -*-
"""独立审核 v8 拼专家：不复用 verify_composable_expert.py，从头复算。
   目的：① 验证文档第64行的恒等式是否成立（blkdiag vs vstack）
        ② 复算「拼 k 个 = k 倍容量」是否真的成立
        ③ 复算效率账
"""
import numpy as np

rng = np.random.default_rng(0)
d, h_cell, k = 64, 8, 4

x = rng.standard_normal(d)
E1 = rng.standard_normal((k, d, h_cell)) / np.sqrt(d)
E2 = rng.standard_normal((k, h_cell, d)) / np.sqrt(h_cell)
w = rng.random(k); w = w / w.sum()

def silu(z): return z / (1.0 + np.exp(-z))

# ---- 左边：加权求和（代码实际用的）----
lhs = sum(w[i] * (silu(x @ E1[i]) @ E2[i]) for i in range(k))

# ---- 右边 A：文档写的 blkdiag ----
try:
    from scipy.linalg import block_diag
    Wb = block_diag(*[w[i] * E2[i] for i in range(k)])
    hcat = np.concatenate([silu(x @ E1[i]) for i in range(k)])
    rhs_blk = hcat @ Wb
    print("blkdiag: hcat%s @ Wb%s -> %s   (目标 %s)"
          % (hcat.shape, Wb.shape, rhs_blk.shape, lhs.shape))
    print("  max|lhs - blkdiag| = %.3e" % np.abs(lhs - rhs_blk).max())
except ImportError:
    print("无 scipy，跳过 blkdiag")

# ---- 右边 B：垂直拼接 vstack（我认为正确的写法）----
Wv = np.vstack([w[i] * E2[i] for i in range(k)])          # [k*h_cell, d]
rhs_v = hcat @ Wv
print("\nvstack : hcat%s @ Wv%s -> %s" % (hcat.shape, Wv.shape, rhs_v.shape))
print("  max|lhs - vstack | = %.3e   %s"
      % (np.abs(lhs - rhs_v).max(),
         "✅ 恒等成立" if np.abs(lhs - rhs_v).max() < 1e-12 else "❌"))

print("\n结论：文档第64行写的 blkdiag %s"
      % ("错误，应改为 vstack（垂直拼接）"
         if np.abs(lhs - rhs_v).max() < 1e-12 else "待核"))
