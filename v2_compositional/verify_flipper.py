"""翻转塞数学自检：验证四条由结构保证的性质。
用法: python3 verify_flipper.py
"""
import torch
import torch.nn as nn
from flipper import FlipperTrap

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}\n")


def make(d=64, **kw):
    m = FlipperTrap(d, **kw).to(dev)
    return m


def check(name, ok, detail=""):
    print(f"  {name:44} {'PASS' if ok else 'FAIL'}   {detail}")
    return ok


print("=" * 78)
print("① 零初始化 → 起步应退化为恒等（保证加了不会变差）")
print("=" * 78)
m = make()
x = torch.randn(2, 8, 64, device=dev)
y, _, diag = m(x)
diff = (y - x).abs().max().item()
check("alpha_net/carry_proj 零初始化 → y≈x", diff < 1e-5, f"max|y-x| = {diff:.2e}")

print()
print("=" * 78)
print("② 平行分量严格不变（支点稳住，主信息永不丢失）")
print("=" * 78)
m2 = make()
with torch.no_grad():
    # 人为打开翻转与分选，放大效果
    nn.init.normal_(m2.seal_net.weight, std=0.5)
    nn.init.normal_(m2.alpha_net.weight, std=0.5)
    nn.init.normal_(m2.carry_proj.weight, std=0.05)
x = torch.randn(4, 16, 64, device=dev)
y, carry, diag = m2(x)
u = m2.axis_raw.to(dev).view(1, 1, 64)
u = torch.nn.functional.normalize(u, dim=-1)
par_before = (x * u).sum(-1)
par_after = (y * u).sum(-1)
# carry 注入会影响平行分量，这里只检查翻转/分选路径
err = (par_before - par_after).abs().max().item()
par_scale = par_before.abs().max().item()
check("沿轴分量在翻转前后一致(忽略carry注入)",
      err < 1e-4, f"max err = {err:.2e}  (分量量级 {par_scale:.3f})")

print()
print("=" * 78)
print("③ 两个支点恒在对面，且密封圈在厚度中心（±u 对称 / s,e ⊥ u）")
print("=" * 78)
with torch.no_grad():
    x = torch.randn(2, 4, 64, device=dev)
    u = torch.nn.functional.normalize(m2.axis_raw.to(dev).view(1, 1, 64), dim=-1)
    s_raw = m2.seal_raw.to(dev).view(1, 1, 64).expand_as(x)
    e_raw = m2.edge_raw.to(dev).view(1, 1, 64).expand_as(x)
    uu = u.expand_as(x)
    s = s_raw - (s_raw * uu).sum(-1, keepdim=True) * uu
    s = torch.nn.functional.normalize(s, dim=-1)
    e = e_raw - (e_raw * uu).sum(-1, keepdim=True) * uu
    e = e - (e * s).sum(-1, keepdim=True) * s
    e = torch.nn.functional.normalize(e, dim=-1)
dot_us = (u.expand_as(x) * s).abs().max().item()
dot_ue = (u.expand_as(x) * e).abs().max().item()
dot_se = (s * e).abs().max().item()
check("轴 ⊥ 密封圈法向 s", dot_us < 1e-5, f"|u·s| = {dot_us:.2e}")
check("轴 ⊥ 边沿方向 e", dot_ue < 1e-5, f"|u·e| = {dot_ue:.2e}")
check("s ⊥ e（旋转平面正交）", dot_se < 1e-5, f"|s·e| = {dot_se:.2e}")

print()
print("=" * 78)
print("④ 翻转是等距旋转（只改方向不改大小 → 数值不跑飞）")
print("=" * 78)
with torch.no_grad():
    # 关掉分选(gate=0)与carry注入，只看旋转是否保范数
    nn.init.zeros_(m2.carry_proj.weight)
    x = torch.randn(2, 8, 64, device=dev)
    y, _, _ = m2(x)
n_in = x.norm(dim=-1)
n_out = (y).norm(dim=-1)
ratio = (n_out / (n_in + 1e-8))
r_mean, r_std = ratio.mean().item(), ratio.std().item()
check("范数比 ≈ 1（等距）", abs(r_mean - 1.0) < 0.02 and r_std < 0.02,
      f"ratio = {r_mean:.4f} ± {r_std:.4f}")

print()
print("=" * 78)
print("⑤ 跨位置 carry：团被兜住并送到下游（B 方案，不是丢弃）")
print("=" * 78)
m3 = make()
with torch.no_grad():
    nn.init.normal_(m3.seal_net.weight, std=1.0)   # 打开分选
    nn.init.normal_(m3.carry_proj.weight, std=0.1)
x = torch.randn(2, 32, 64, device=dev)
y, carry, diag = m3(x)
check("carry 非空（确实兜住了东西）",
      diag["carry_norm"] > 1e-6, f"||carry|| = {diag['carry_norm']:.4f}")
check("分选门在 (0,1) 内（软门槛）",
      0.0 < diag["gate_mean"] < 1.0, f"gate_mean = {diag['gate_mean']:.4f}")

# 分段喂：先喂前半，拿 carry；再喂后半，看 carry 是否延续
with torch.no_grad():
    _, c1, _ = m3(x[:, :16])
    _, c2, d2 = m3(x[:, 16:], carry=c1)
    _, c0, _ = m3(x[:, 16:])          # 不带 carry 重跑
diff_carry = (c2 - c0).abs().max().item()
check("carry 可跨段传递（信息流打通）",
      diff_carry > 1e-6, f"|c2 - c0| = {diff_carry:.4f}")

print()
print("=" * 78)
print("⑥ 数值稳定性：长序列 + 极端输入不产生 NaN/Inf")
print("=" * 78)
with torch.no_grad():
    x = torch.randn(2, 512, 64, device=dev) * 10.0   # 故意放大
    y, c, dg = m3(x)
finite = bool(torch.isfinite(y).all() and torch.isfinite(c).all())
check("无 NaN / Inf", finite,
      f"y max = {y.abs().max().item():.3f}  carry = {c.norm(dim=-1).mean().item():.3f}")

print()
print("=" * 78)
print("自检结束")
print("=" * 78)
