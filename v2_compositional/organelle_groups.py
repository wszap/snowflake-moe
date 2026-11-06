"""软群：把 [C,n] 的 logits 分解成 [C,G]×[G,n]，全 n 个忆点权重恒非零。
禁止选群 / topk / onehot —— 群只是低秩结构先验，不是路由。
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class SoftGroups(nn.Module):
    def __init__(self, d, n, G=8, tau=0.5, temp_g=2.0, ent_floor=0.5):
        super().__init__()
        self.d, self.n, self.G = d, n, G
        self.tau, self.temp_g = tau, temp_g
        self.ent_floor = ent_floor
        self.P = nn.Parameter(F.normalize(torch.randn(G, d), dim=-1))

    def membership(self, M, center=True):
        """M [n,d] → A [n,G] 软归属。center=True 去掉共享 base 主簇，露出个体差异。"""
        Mn = F.normalize(M, dim=-1)
        Pn = F.normalize(self.P, dim=-1)
        if center:
            Mn = F.normalize(Mn - Mn.mean(0, keepdim=True), dim=-1)
            Pn = F.normalize(Pn - Pn.mean(0, keepdim=True), dim=-1)
        return F.softmax(Mn @ Pn.T / self.tau, dim=-1)

    @torch.no_grad()
    def init_from_memories(self, M, iters=20):
        """从忆点【中心化方向】软 k-means 长出群质心。群是长出来的，不是随机撒的。"""
        Mn = F.normalize(M.detach(), dim=-1)
        X  = F.normalize(Mn - Mn.mean(0, keepdim=True), dim=-1)   # 去共享 base
        n, G = X.shape[0], self.G
        C = X[torch.randperm(n, device=X.device)[:G]].clone()
        for _ in range(iters):
            A = F.softmax(X @ C.T / self.tau, dim=-1)             # [n,G] 软归属
            C = F.normalize(A.T @ X, dim=-1)                       # [G,d] 软质心
        self.P.copy_(C)
        return self

    def wiring(self, xsum, M):
        """xsum [C,d] → w [C,n]，全非零；同时返回 w_g 供熵监控"""
        A = self.membership(M)                                   # [n,G]
        w_g = F.softmax(xsum @ F.normalize(self.P, dim=-1).T
                        / math.sqrt(self.d) * self.temp_g, dim=-1)   # [C,G]
        w = w_g @ A.T                                            # [C,n]
        w = w / w.sum(-1, keepdim=True)
        return w, w_g

    def compose_prior(self, w_orig, xsum, M, beta=0.1):
        """群是【乘性结构先验】，不是路由：保住 w_orig 尖锐度，全 n 权重恒非零（仍是拼）"""
        A = self.membership(M)                                       # [n,G]
        Pn = F.normalize(self.P, dim=-1)
        w_g = F.softmax(xsum @ Pn.T / math.sqrt(self.d) * self.temp_g, dim=-1)  # [C,G]
        w_struct = w_g @ A.T                                           # [C,n]
        w = w_orig * (1.0 + beta * w_struct)
        return w / w.sum(-1, keepdim=True), w_g

    @torch.no_grad()
    def health(self, M):
        """判据：全非零 / 群未塌缩 / 熵有下界"""
        _, w_g = self.wiring(F.normalize(M, dim=-1)[:16], M)
        ent = -(w_g * (w_g + 1e-9).log()).sum(-1).mean() / math.log(self.G)
        Pn = F.normalize(self.P, dim=-1)
        _G = Pn @ Pn.T
        collapse = _G[~torch.eye(self.G, dtype=torch.bool, device=_G.device)].abs().max().item()
        return dict(entropy_ratio=ent.item(), P_max_cos=collapse,
                    eff_lower_bound=1.0 / (self.wiring(F.normalize(M, dim=-1)[:16], M)[0]
                                          .pow(2).sum(-1).mean()).item())
