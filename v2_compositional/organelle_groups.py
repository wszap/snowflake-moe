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

    def membership(self, M):
        """M [n,d] 忆点 → A [n,G] 软归属（形态相似/功能相同 = 软聚类）"""
        return F.softmax(F.normalize(M, dim=-1) @ F.normalize(self.P, dim=-1).T / self.tau, dim=-1)

    def wiring(self, xsum, M):
        """xsum [C,d] → w [C,n]，全非零；同时返回 w_g 供熵监控"""
        A = self.membership(M)                                   # [n,G]
        w_g = F.softmax(xsum @ F.normalize(self.P, dim=-1).T
                        / math.sqrt(self.d) * self.temp_g, dim=-1)   # [C,G]
        w = w_g @ A.T                                            # [C,n]
        w = w / w.sum(-1, keepdim=True)
        return w, w_g

    @torch.no_grad()
    def health(self, M):
        """判据：全非零 / 群未塌缩 / 熵有下界"""
        _, w_g = self.wiring(F.normalize(M, dim=-1)[:16], M)
        ent = -(w_g * (w_g + 1e-9).log()).sum(-1).mean() / math.log(self.G)
        Pn = F.normalize(self.P, dim=-1)
        collapse = (Pn @ Pn.T).off_diagonal().abs().max().item()
        return dict(entropy_ratio=ent.item(), P_max_cos=collapse,
                    eff_lower_bound=1.0 / (self.wiring(F.normalize(M, dim=-1)[:16], M)[0]
                                          .pow(2).sum(-1).mean()).item())
