"""主列表库：主动登记/分裂/接线/复用/写分工。
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
        self.in_centroid = z(n_max, d)       # 功能：它「对什么」响应
        self.in_var      = z(n_max)          # 通用范围：输入分布宽度
        self.fitness     = z(n_max)          # 被选中强度
        self.teammates   = z(n_max, 8)       # 主要队友
        self.hit_input   = z(n_max, d)       # ★ 自身被选中时的输入中心（主动接线信号）
        self.group_id    = torch.full((n_max,), -1, device=self.dev, dtype=torch.long)
        self.birth_step  = z(n_max)
        self.parent_id   = torch.full((n_max,), -1, device=self.dev, dtype=torch.long)
        self.w_child     = z(n_max)          # 出生/接线份额（主动设定，非梯度学）
        self.active      = torch.zeros(n_max, device=self.dev, dtype=torch.bool)
        self.active[:n_init] = True

    def support(self, i):
        """通用范围：in_var / ||in_centroid||。越宽 = 被越多不同输入激活 = 该分裂了。"""
        return (self.in_var[i] / (self.in_centroid[i].norm() + 1e-9)).item()

    @torch.no_grad()
    def observe(self, xsum, w, step):
        """主动写分工：功能=响应中心，范围=分布宽度，队友=共激活，接线=自身命中输入"""
        m, v = xsum.mean(0), xsum.var(0).mean()
        act = w.mean(0)                                   # [n]
        top = torch.topk(w, 8, dim=-1).indices            # [C,8]
        nn  = w.shape[-1]
        for i in range(min(self.n, nn)):
            if not self.active[i]: continue
            mask = (top == i).any(-1)                     # ★ 先定义，下面两处共用
            self.in_centroid[i].mul_(self.ema).add_(m * (1 - self.ema))
            self.in_var[i].mul_(self.ema).add_(v * (1 - self.ema))
            self.fitness[i].mul_(self.ema).add_(act[i] * (1 - self.ema))
            if mask.any():
                self.hit_input[i].mul_(self.ema).add_(xsum[mask].mean(0) * (1 - self.ema))
                co = torch.bincount(top[mask].flatten(), minlength=nn)[:nn].float()
                co[i] = 0
                self.teammates[i] = torch.topk(co, 8).values / (co.sum() + 1e-9)
        return self

    def should_divide(self, i, step, min_gap=200):
        """不能无限长大：support 变宽 + 忙 → 生一个，绝不扩 rank（非 step 时钟）"""
        if not self.active[i] or self.n >= self.n_max: return False
        if step - self.birth_step[i].item() < min_gap: return False
        return self.support(i) > self.tau_support and self.fitness[i].item() > self.tau_fitness

    def divide(self, parent, mem, step, eps=0.3, group=-1):
        """继承「领地」，不继承「知识」——W1_U/W2_U 由调用方零初始化"""
        j = self.n
        child_mem = F.normalize(mem[parent] + eps * torch.randn(self.d, device=mem.device), dim=0)
        self.in_centroid[j] = self.in_centroid[parent].clone()
        self.hit_input[j]   = self.in_centroid[parent].clone()   # ★ 出生即有接线信号
        self.in_var[j] = 0.0; self.fitness[j] = 0.0
        self.group_id[j]   = group if group >= 0 else self.group_id[parent]
        self.birth_step[j] = step; self.parent_id[j] = parent
        self.w_child[j]    = 0.0017
        self.active[j] = True; self.n += 1
        return j, child_mem

    @torch.no_grad()
    def wire_in(self, j, xsum, mem, eta=0.02, cap=0.05):
        """主动接线：份额由【自身忆点 × 自身命中输入】的命中率驱动，非日程、非梯度"""
        q = F.normalize(self.hit_input[j], dim=0)
        hit = (F.normalize(mem[j], dim=0) @ q).item()
        hit = max(0.0, min(1.0, (hit + 1) / 2))
        self.w_child[j] = min(cap, self.w_child[j] + eta * hit)
        return self.w_child[j].item()

    @torch.no_grad()
    def recruit(self, q, mem, k=8):
        """主动复用：全局池，谁忆点匹配谁被招募"""
        s = F.normalize(mem[:self.n], dim=-1) @ F.normalize(q, dim=0)
        return torch.topk(s, min(k, self.n)).indices.tolist()

    def card(self, i):
        return dict(id=i, group=int(self.group_id[i]), parent=int(self.parent_id[i]),
                    birth=int(self.birth_step[i]), fitness=round(self.fitness[i].item(), 4),
                    support=round(self.support(i), 4), w_child=round(self.w_child[i].item(), 5),
                    teammates=[round(v, 3) for v in self.teammates[i].tolist()])
