"""C1 决策层：主动分裂 / 冻结 key / 秩-1 群内接线 / 加性份额。
不改任何现有源码。cells = [SnowflakeB, ...]（LM 里所有 cell，或冒烟用单个）。
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from organelle_registry import OrganelleRegistry
from organelle_groups import SoftGroups


class SplitRuntime:
    def __init__(self, cells, d, n_init, K=8, G=8, tau=0.10,
                 split_at=1500, eps=0.3, seed=0):
        self.cells, self.d, self.n0 = list(cells), d, n_init
        self.K, self.G, self.split_at = K, G, split_at
        self.eps, self.step_count, self.J = eps, 0, 0
        self.wired = False

        M0 = self.cells[0].organelle_memory.detach()
        self.reg = OrganelleRegistry(d, n_init, n_max=n_init + K)
        self.groups = SoftGroups(d, n_init, G=G, tau=tau)
        self.groups.init_from_memories(M0)
        gid = self.groups.membership(M0).argmax(-1)
        self.reg.group_id[:n_init] = gid
        self.peers = [(gid == g).nonzero().flatten() for g in range(G)]
        self.child_group = []                      # 每个新细胞所属群
        self.child_w = torch.zeros(K)              # 主动份额（非梯度学）

        # 冻结 connect / key：c 慢变量，分裂前后旧 key 严格不变
        self.c_cache, self.key_cache = [], []
        for c in self.cells:
            cc, _ = c.connect()
            M = c.organelle_memory.detach()
            self.c_cache.append(cc.detach().clone())
            self.key_cache.append((M + cc.T @ M).clone())

    # ---- 主动触发：support/fitness，非 step 时钟 ----
    def observe(self, xsum, w_old, step):
        self.reg.observe(xsum, w_old[:, :self.n0], step)
        return self

    def maybe_divide(self, step, xsum, force=False):
        if self.J >= self.K: return 0
        if not force and step < self.split_at: return 0
        fired = 0
        for i in range(self.n0):
            if self.J >= self.K: break
            if force or self.reg.should_divide(i, step):
                parent = i
                g = int(self.reg.group_id[parent])
                j, nm = self.reg.divide(parent, self.cells[0].organelle_memory.detach(),
                                        step, eps=self.eps, group=g)
                self.child_group.append(g)
                self.child_w[self.J] = self.reg.w_child[j].item()
                peers = self.peers[g]
                for ci, c in enumerate(self.cells):
                    self._expand_one(c, ci, parent, peers, step)
                self.J += 1; fired += 1
        return fired

    def _expand_one(self, c, ci, parent, peers, step):
        """扩 5 个按 n 索引的张量 + 零补 c + key_cache 追加一行（隔离期严格不扰动）"""
        n, d = self.n0 + self.J, self.d
        M = c.organelle_memory.detach()
        nm = F.normalize(M[parent] + self.eps * torch.randn(d), dim=0).unsqueeze(0)
        r = c.W1_U.shape[-1]
        with torch.no_grad():
            c.organelle_memory = nn.Parameter(torch.cat([M, nm], 0))
            c.W1_U = nn.Parameter(torch.cat([c.W1_U.data, torch.zeros(1, d, r)], 0))
            h = c.W2_U.shape[1]
            c.W2_U = nn.Parameter(torch.cat([c.W2_U.data, torch.zeros(1, h, r)], 0))
            c.register_buffer('mix_ema', torch.cat([c.mix_ema, torch.zeros(1)]))
            c.register_buffer('fitness_ema', torch.cat([c.fitness_ema, torch.zeros(1)]))
            c.n = n + 1
            # Schur 零补：旧块原封不动，新行列出生成隔离期
            cc = torch.zeros(n + 1, n + 1)
            cc[:n, :n] = self.c_cache[ci]
            self.c_cache[ci] = cc
            # 隔离期 key_new[n] = M_new[n]（step14b 已证），旧 n 行严格不变
            self.key_cache[ci] = torch.cat([self.key_cache[ci], nm], 0)

    # ---- 主动接线：份额由自身忆点命中率；结构用秩-1 只落本群 ----
    def wire(self, xsum):
        if self.J == 0: return self.child_w[:0]
        for k in range(self.J):
            j = self.n0 + k
            self.child_w[k] = self.reg.wire_in(j, xsum, self.cells[0].organelle_memory,
                                              eta=0.02, cap=0.05)
        if not self.wired:
            for k in range(self.J):
                n, g = self.n0 + k, self.child_group[k]
                peers = self.peers[g]
                for ci in range(len(self.cells)):
                    cc = self.c_cache[ci]
                    cc[n, peers] = 1.0 / len(peers)
                    cc[n] /= cc[n].sum()
                    nm = self.key_cache[ci][n]
                    # ★ 秩-1：只有 peers 行变，群外严格 0
                    self.key_cache[ci][:self.n0] += cc[n, :self.n0].unsqueeze(-1) * nm
            self.wired = True
        return self.child_w[:self.J].clone()

    # ---- 拼专家：前 n0 维冻结 w_old，后 J 维主动份额，全非零 ----
    def child_fm(self, w_old):
        if self.J == 0: return w_old
        cols = self.child_w[:self.J].to(w_old).unsqueeze(0).expand(w_old.shape[0], -1)
        return torch.cat([w_old, cols], -1)

    def status(self):
        return dict(step=self.step_count, divided=self.J, K=self.K,
                    wired=self.wired, child_w=[round(v, 5) for v in self.child_w[:self.J].tolist()],
                    groups=[int(g) for g in self.child_group])
