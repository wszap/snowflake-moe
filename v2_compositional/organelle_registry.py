"""细胞主列表库：主动登记 / 主动分裂 / 主动接线 / 主动复用 / 主动写分工。
不复制母体已学知识；群里共享的只有标签与质心，不是突触。
"""
import torch, torch.nn.functional as F

class OrganelleRegistry:
    def __init__(self, d, n_init, n_max=1024, device="cpu",
                 tau_support=0.70, tau_fitness=0.010, ema=0.99):
        self.d, self.n, self.n_max = d, n_init, n_max
        self.dev = torch.device(device)
        self.tau_support, self.tau_fitness, self.ema = tau_support, tau_fitness, ema
        z = lambda *s: torch.zeros(*s, device=self.dev)
        self.in_centroid = z(n_max, d)      # 它「对什么」响应（功能）
        self.in_var      = z(n_max)         # 分布宽度 → 通用范围
        self.fitness     = z(n_max)         # 被选中强度
        self.teammates   = z(n_max, 8)      # 主要队友（top8）
        self.group_id    = torch.full((n_max,), -1, device=self.dev, dtype=torch.long)
        self.birth_step  = z(n_max)
        self.parent_id   = torch.full((n_max,), -1, device=self.dev, dtype=torch.long)
        self.w_child     = z(n_max)         # 出生/接线份额（主动设定，非梯度学）
        self.active      = torch.zeros(n_max, device=self.dev, dtype=torch.bool)
        self.active[:n_init] = True

    # ---- 主动写分工：每步自己累积，不是事后解释 ----
    @torch.no_grad()
    def observe(self, xsum, w, step):
        """xsum [C,d] 当前 chunk 表征；w [C,n] 本步 wiring"""
        m = xsum.mean(0)
        v = xsum.var(0).mean()
        act = w.mean(0)                                  # [n]
        idx = torch.topk(w, 8, dim=-1).indices
        for i in range(self.n):
            if not self.active[i]: continue
            a = act[i].item()
            self.in_centroid[i].mul_(self.ema).add_(m * (1 - self.ema))
            self.in_var[i].mul_(self.ema).add_(v * (1 - self.ema))
            self.fitness[i].mul_(self.ema).add_(a * (1 - self.ema))
            cnt = torch.bincount(idx[:, :8].flatten(), minlength=self.n)[:self.n].float()
            self.teammates[i] = torch.topk(cnt, 8).values / (cnt.sum() + 1e-9)
        return self

    # ---- 不能无限长大：support 变宽就生一个，不扩 rank ----
    def should_divide(self, i, step, min_gap=200):
        if not self.active[i] or self.n >= self.n_max: return False
        if step - self.birth_step[i].item() < min_gap: return False
        support = (self.in_var[i] / (self.in_centroid[i].norm() + 1e-9)).item()
        return (support > self.tau_support
                and self.fitness[i].item() > self.tau_fitness)

    # ---- 主动分裂：继承「领地」，不继承「知识」 ----
    def divide(self, parent, mem, step, eps=0.3, group=-1):
        j = self.n
        child_mem = F.normalize(mem[parent] + eps * torch.randn(self.d, device=mem.device), dim=0)
        self.in_centroid[j]  = self.in_centroid[parent].clone()
        self.in_var[j]       = 0.0
        self.fitness[j]      = 0.0
        self.group_id[j]     = group if group >= 0 else self.group_id[parent]
        self.birth_step[j]   = step
        self.parent_id[j]    = parent
        self.w_child[j]      = 0.0017          # bias≈-5 的等价份额：Δ≈0，梯度非零
        self.active[j]       = True
        self.n += 1
        return j, child_mem                    # W1_U/W2_U 由调用方零初始化

    # ---- 主动接线：份额由忆点命中率驱动，不由梯度、不由日程 ----
    @torch.no_grad()
    def wire_in(self, j, xsum, mem, eta=0.02, cap=0.05):
        q = F.normalize(xsum.mean(0), dim=0)
        hit = (F.normalize(mem[:self.n], dim=-1) @ q).mean().item()
        hit = max(0.0, min(1.0, (hit + 1) / 2))
        self.w_child[j] = min(cap, self.w_child[j] + eta * hit)
        return self.w_child[j].item()

    # ---- 主动复用：全局池，谁的忆点匹配谁就能招募 ----
    @torch.no_grad()
    def recruit(self, q, mem, k=8):
        s = F.normalize(mem[:self.n], dim=-1) @ F.normalize(q, dim=0)
        return torch.topk(s, min(k, self.n)).indices.tolist()

    # ---- 分工卡：可审计导出 ----
    def card(self, i):
        return dict(id=i, group=int(self.group_id[i]),
                    parent=int(self.parent_id[i]),
                    birth=int(self.birth_step[i]),
                    fitness=round(self.fitness[i].item(), 4),
                    support=round((self.in_var[i] / (self.in_centroid[i].norm() + 1e-9)).item(), 4),
                    w_child=round(self.w_child[i].item(), 5),
                    centroid_norm=round(self.in_centroid[i].norm().item(), 4),
                    teammates=[round(v, 3) for v in self.teammates[i].tolist()])
