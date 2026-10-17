"""
=============================================================================
Snowflake 架构 v3 —— 补完件：核架 Scaffold（先筛选再给）+ 安全机制接口
=============================================================================

你的完整思想（按你确认的顺序）：

    输入
      ↓
    核架 ① 全文关联（收集）
      ↓
    核架 ② 选择门：筛出「最符合本次要求」的信息   ← ★ 本次新增
      ↓
    忆点：接收核架【筛选后】的上下文；忆点之间互相串联
      ↓
    细胞器：无路由，自发分工（谁干多少自己决定）
      ↓
    忆点：整理细胞器输出
      ↓
    安全机制：防自我操控 / 需人同意 / 符合安全与国际要求   ← 策略待你定
      ↓
    输出

=============================================================================
核架为什么是「两级」而不是一层注意力
=============================================================================
    第一级「收集」：因果全文关联 —— 让每个位置都看见自己的过去
    第二级「筛选」：按当前位置的需求，从过去里挑出真正相关的若干条

    区别：
        普通注意力 = 过去全部参与，只是权重不同（软）
        核架筛选   = 只有被选中的 k 条参与，其余权重严格为 0（硬稀疏）

    这也是核架与 dense baseline 的本质区别 —— 不是"有没有上下文"，
    而是"上下文是全收还是先用再筛"。

=============================================================================
集成要点
=============================================================================
1. chunk_size 改回 1 —— 撤掉 chunk 平均（泄漏源）
   chunk_size<=1 时 _chunk_view 返回 x.unsqueeze(1)，xc.mean(1) == x 本身

2. 每层改成残差（原代码 x = sum(outs)/len(outs) 无残差）：
       x = x + Scaffold(x)      # 核架：关联 + 筛选
       x = x + Organelles(x)    # 忆点+细胞器：替换 FFN

3. 补位置编码（原设计缺失；dense 为公平也主动放弃了，所以两边都该补）

4. 安全机制放在最后一层之后、head 之前 —— 细胞器绕不开
   注意：这是【占位实现】，策略阈值由人设定，模型不自证合规

=============================================================================
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 一、核架 Scaffold —— 先关联全文，再筛选，最后交给忆点
# ============================================================================
class Scaffold(nn.Module):
    """核架：收集并整理「最符合当前请求」的信息，交给忆点。

    数据流（两级）：
        ① 收集 collect: 因果自注意力 —— 每个位置看见自己的过去
        ② 筛选 select : 选择门 —— 按当前位置的需求，只保留最相关的 top_k 条
        ③ 整理 output : 对被选中的 k 条加权求和，交给忆点

    职责边界（重要）：
        - 核架【只负责】跨 token 的信息流（上下文）
        - 核架【不负责】指派细胞器（那是忆点的事，没有 router）

    ⚠️ 因果性在两级都要保证：
        ① 收集用 is_causal=True（不要手写 triu mask 再传，
           之前 dense_baseline 就是这么写反的：mask 里 True = 参与，
           导致模型只看未来，loss 崩到 0.019）
        ② 筛选时对未来位置填 -inf，top_k 只会从 0..t 里选
    """

    def __init__(self, d: int, n_head: int, top_k: int = 32,
                 dropout: float = 0.0, select_mode: str = "topk"):
        super().__init__()
        assert d % n_head == 0, f"d={d} 必须能被 n_head={n_head} 整除"
        assert select_mode in ("topk", "soft", "none"), \
            f"select_mode 必须是 topk/soft/none，收到 {select_mode}"

        self.d = d
        self.n_head = n_head
        self.head_dim = d // n_head
        self.top_k = top_k
        self.select_mode = select_mode

        # ---- ① 收集：因果全文关联 ----
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

        # ---- ② 筛选：选择门（"我要什么" vs "每个位置有什么"）----
        self.norm2 = nn.LayerNorm(d)
        self.gate_q = nn.Linear(d, d, bias=False)   # 当前位置的需求
        self.gate_k = nn.Linear(d, d, bias=False)   # 各历史位置的内容
        self.gate_out = nn.Linear(d, d, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, T, d] → [B, T, d]（筛选并整理后的上下文，交给忆点）"""
        B, T, d = x.shape

        # ========== ① 收集：因果全文关联 ==========
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)             # [3, B, H, T, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # ★ 因果：位置 t 只能看 0..t
        ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, d)
        ctx = self.drop(self.proj(ctx))

        # 关联后的表示（残差）—— 后续筛选在它上面做
        h2 = self.norm2(x + ctx)

        # ---- 消融开关：不筛选时直接返回关联结果（用于证明筛选的价值）----
        if self.select_mode == "none":
            return ctx

        # ========== ② 筛选：当前位置的需求 vs 各历史位置 ==========
        gq = self.gate_q(h2)                          # [B, T, d]
        gk = self.gate_k(h2)                          # [B, T, d]
        s = torch.matmul(gq, gk.transpose(1, 2)) / math.sqrt(d)   # [B, T, T]

        # ★ 因果：不许选未来（位置 i 只能选 j <= i）
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        s = s.masked_fill(causal, float("-inf"))

        if self.select_mode == "soft":
            # 软：全部历史参与，只是权重不同（等价于普通注意力，用于对照）
            w = torch.softmax(s, dim=-1)
            out = torch.matmul(w, h2)
            return self.drop(self.gate_out(out))

        # ---- topk：真稀疏，只有被选中的 k 条参与，其余权重严格为 0 ----
        k = min(self.top_k, T)
        topv, topi = torch.topk(s, k=k, dim=-1)       # [B, T, k]

        # ★ 关键修复（已用 numpy 复现验证：修复前 128 行共 496 条越界）
        #   当 t < k-1 时，合法历史不足 k 条，topk 会把被屏蔽(-inf)的
        #   未来位置也选进来。它们的 softmax 权重是 0，【数值上】不影响输出，
        #   但 gather 会把计算图连到未来 —— 一旦后续加直通残差或改动门控，
        #   就会立刻变成真泄漏。这里把索引 clamp 到 [0, t] 彻底断开：
        #   被 clamp 的位置权重恰为 0，所以 0 × h2[t] 与 0 × h2[j] 完全等价，
        #   数值不变，但计算图不再触及未来。
        arange_t = torch.arange(T, device=x.device).view(1, T, 1)
        topi = torch.minimum(topi, arange_t)           # 索引永不指向未来

        # 只在被选中的 k 条上做 softmax（其余为 0，梯度也只从这 k 条回传）
        attn_w = torch.softmax(topv, dim=-1)           # [B, T, k]

        # 取出被选中位置的表示
        idx = topi.unsqueeze(-1).expand(B, T, k, d)                # [B,T,k,d]
        src = h2.unsqueeze(1).expand(B, T, T, d)                   # [B,T,T,d]
        v_sel = torch.gather(src, 2, idx)                          # [B,T,k,d]

        out = (attn_w.unsqueeze(-1) * v_sel).sum(dim=2)            # [B, T, d]

        return self.drop(self.gate_out(out))


# ============================================================================
# 二、安全机制 SafetyGate —— 治理层接口（策略待定，先留位置）
# ============================================================================
class SafetyGate(nn.Module):
    """安全机制（治理层接口）

    ========================================================================
    架构能做到的三件事
    ========================================================================
      1. 不可绕过 —— 门在所有细胞器之后、head 之前。
         细胞器的 wiring 只决定内部计算，无法绕开门。
      2. 可审计   —— 记录哪些忆点/细胞器被激活。
         稀疏激活是天然优势：每个输出都能回答「为什么」。
      3. 可拒绝   —— 超出授权范围时走拒绝/降级通路，不产生输出。

    ========================================================================
    架构【做不到】的，必须在系统层落实（不要过度承诺）
    ========================================================================
      - 判断什么算「高风险」：策略来自人，不能由模型自己定义
      - 真正拦住执行：模型只能「提议」，执行权必须在人手里
      - 合规认证：需要外部审计与流程，架构无法自证

    ⚠️ 你说过这一层「还没想好」—— 所以这里是【占位实现】：
       接口、位置、审计记录都留好了，但判定策略留空等你定。
       不要指望架构保证安全，那是过度承诺。

    ========================================================================
    对应国际要求（EU AI Act 思路）
    ========================================================================
      透明度/可追溯 → 忆点激活记录供解释
      人类监督     → 高风险输出暂停，交人确认后放行
      无自主目标   → 推理期参数冻结，不可自我修改/自设目标
      鲁棒性       → 拒绝/降级通路
    """

    def __init__(self, d: int, risk_threshold: float = 0.5):
        super().__init__()
        self.risk_threshold = risk_threshold
        self.risk_head = nn.Linear(d, 1)      # 风险打分（阈值由人设定）

    def forward(self, x, wiring=None, organelle_ids=None):
        """x: [B, T, d] → (x, info)

        info 里带出：
            risk                 —— 风险分
            needs_authorization  —— 是否需人工确认（暂停，不直接放行）
            out_of_scope         —— 是否超出授权范围（走拒绝通路）
            audit                —— 审计记录：哪些忆点/细胞器被激活
        """
        pooled = x.mean(dim=1)                       # [B, d]
        risk = torch.sigmoid(self.risk_head(pooled))  # [B, 1]

        audit = {}
        if wiring is not None:
            # 稀疏激活 → 可追溯：记录被显著激活的忆点
            w = wiring.detach()
            topv, topi = torch.topk(w, k=min(8, w.shape[-1]), dim=-1)
            audit["top_memory_points"] = topi.float().mean(0).tolist()
            audit["top_weights"] = topv.float().mean(0).tolist()
            audit["eff_organs"] = float(
                (w.sum(-1) ** 2 / (w ** 2).sum(-1)).mean().item()
            )
        if organelle_ids is not None:
            audit["organelle_ids"] = organelle_ids

        info = {
            "risk": risk.detach(),
            "needs_authorization": bool((risk.max().item()) > self.risk_threshold),
            "out_of_scope": False,   # 由外部策略层判定，模型不自证
            "audit": audit,
        }
        return x, info


# ============================================================================
# 三、v3 完整模型：核架(关联+筛选) → 忆点 → 细胞器 → 安全机制
# ============================================================================
class SnowflakeLMV3(nn.Module):
    """集成示意。SnowflakeB 直接复用现有实现（传 chunk_size=1 以消除泄漏）。

    与 v2 的差别：
        + 核架 Scaffold（关联 + 先筛选再给忆点）  ← 原设计有、代码漏了
        + 位置编码                                  ← 原设计缺
        + 残差连接                                  ← 原代码 sum(outs)/len 无残差
        + 安全机制（占位）                          ← 原设计有、代码漏了
        - chunk 平均（泄漏源）                      ← chunk_size=1 后自动消失
    """

    def __init__(self, vocab_size, d=128, n=384, L=4, h=32, n_cells=4,
                 seq_len=128, rank=4, temp=5.0, n_head=4,
                 hinge_mode="loss", use_hinge_low=False,
                 min_eff_organs=0, seed=0, risk_threshold=0.5,
                 scaffold_top_k=32, select_mode="topk"):
        super().__init__()
        from snowflake_B import SnowflakeB      # 复用现有细胞器实现

        self.d, self.L, self.n_cells = d, L, n_cells
        self.seq_len = seq_len
        self.scaffold_top_k = scaffold_top_k
        self.select_mode = select_mode

        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(seq_len, d)      # ★ 补位置编码
        self.in_proj = nn.Linear(d, d, bias=False)

        # 核架：每层一个（关联 + 筛选）
        self.scaffolds = nn.ModuleList([
            Scaffold(d, n_head=n_head, top_k=scaffold_top_k,
                     select_mode=select_mode)
            for _ in range(L)
        ])

        # 细胞器：chunk_size=1 —— 撤掉 chunk 平均，消除未来泄漏
        self.layers = nn.ModuleList([
            nn.ModuleList([
                SnowflakeB(d, n_organelles=n, h=h, rank=rank,
                           seq_len=seq_len, chunk_size=1,   # ★ 关键
                           temp=temp, hinge_mode=hinge_mode,
                           use_hinge_low=use_hinge_low,
                           log_sharp=True,
                           min_eff_organs=min_eff_organs,
                           seed=seed * 1000 + li * 10 + ci)
                for ci in range(n_cells)
            ]) for li in range(L)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.final_norm = nn.LayerNorm(d)
        self.safety = SafetyGate(d, risk_threshold=risk_threshold)
        self.head = nn.Linear(d, vocab_size)

    def forward(self, tokens, perm=None, force_mix=None):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)

        x = self.in_proj(self.embed(tokens) + self.pos(pos))
        diag_last = {}
        wiring_last = None

        for li, (sc, layer) in enumerate(zip(self.scaffolds, self.layers)):
            # ① 核架：全文因果关联 → 筛出最相关的 → 交给忆点
            ctx = sc(x)
            x = x + ctx

            # ② 细胞器：忆点接收【筛选后】的上下文，自发分工（无 router）
            outs, wiring_last = [], None
            for cell in layer:
                y, info = cell(x.reshape(B * T, self.d),
                               perm=perm, force_mix=force_mix)
                outs.append(y.reshape(B, T, self.d))
                diag_last = info
                if "wiring" in info:
                    wiring_last = info["wiring"]

            # ③ 残差：细胞器替换的是 FFN，所以要加回来
            x = x + sum(outs) / len(outs)
            x = self.norms[li](x)

        # ④ 安全机制：不可绕过 / 可审计 / 可拒绝（策略待你定）
        x, safety_info = self.safety(self.final_norm(x), wiring=wiring_last)
        diag_last["safety"] = safety_info
        diag_last["scaffold_top_k"] = self.scaffold_top_k
        diag_last["scaffold_select_mode"] = self.select_mode

        return self.head(x), diag_last

    def heal_sum(self):
        s = None
        for layer in self.layers:
            for cell in layer:
                if cell.heal_term is not None:
                    s = cell.heal_term if s is None else s + cell.heal_term
        return s


# ============================================================================
# 四、泄漏自检 —— 补完核架后必做
# ============================================================================
@torch.no_grad()
def leak_check(model, vocab_size=100, seq_len=128, B=2, device="cuda"):
    """改动未来位置 16..31 后，位置 0..15 的输出必须零变化。

    有变化 = 未来信息泄漏进了过去 → 数字无效，不能进论文。

    核架有两级，两级都要验：
        ① 收集 is_causal=True
        ② 筛选 对未来填 -inf，top_k 只能从 0..t 里选
    """
    model.eval()
    torch.manual_seed(0)
    tok = torch.randint(0, vocab_size, (B, seq_len), device=device)

    out1, _ = model(tok)

    tok2 = tok.clone()
    tok2[:, 16:32] = torch.randint(0, vocab_size, (B, 16), device=device)
    out2, _ = model(tok2)

    diff = (out1[:, :16] - out2[:, :16]).abs().max().item()
    ok = diff == 0.0
    print(f"  leak_check: 改动位置 16..31 后，位置 0..15 输出最大差异 = {diff:.2e}")
    print(f"  ⇒ {'✅ 无泄漏（正确 causal）' if ok else '❌ 有泄漏！数字不可用于论文'}")
    return ok


# ============================================================================
# 五、自测
# ============================================================================
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}\n")

    print("=== ① Scaffold 单独测试（topk 模式）===")
    sc = Scaffold(d=128, n_head=4, top_k=32, select_mode="topk").to(dev)
    x = torch.randn(2, 128, 128, device=dev)
    y = sc(x)
    print(f"  in {tuple(x.shape)} → out {tuple(y.shape)}")

    # 核架自身不能泄漏（两级都要过）
    x2 = x.clone()
    x2[:, 16:32] = torch.randn_like(x2[:, 16:32])
    d = (sc(x)[:, :16] - sc(x2)[:, :16]).abs().max().item()
    print(f"  核架泄漏检查: 位置 0..15 差异 = {d:.2e}  "
          f"{'✅' if d == 0 else '❌'}")

    print("\n=== ①b 消融对照（soft / none）===")
    for mode in ("soft", "none"):
        sc2 = Scaffold(d=128, n_head=4, top_k=32, select_mode=mode).to(dev)
        y2 = sc2(x)
        d2 = (sc2(x)[:, :16] - sc2(x2)[:, :16]).abs().max().item()
        print(f"  {mode:5s}: out {tuple(y2.shape)}  泄漏={d2:.2e} "
              f"{'✅' if d2 == 0 else '❌'}")

    print("\n=== ② SafetyGate 测试 ===")
    gate = SafetyGate(d=128).to(dev)
    w = F.softmax(torch.randn(8, 384, device=dev), dim=-1)
    _, info = gate(torch.randn(8, 128, 128, device=dev), wiring=w)
    print(f"  needs_authorization = {info['needs_authorization']}")
    print(f"  audit eff_organs    = {info['audit'].get('eff_organs'):.2f}")
    print(f"  audit 记录了 {len(info['audit'].get('top_memory_points', []))} 个忆点")

    print("\n=== ③ 完整模型（需 snowflake_B.py 在同目录）===")
    try:
        m = SnowflakeLMV3(vocab_size=100, d=128, n=384, L=2,
                          n_cells=2, seq_len=128).to(dev)
        p = sum(q.numel() for q in m.parameters())
        print(f"  参数量 = {p:,}")
        print("\n=== ④ 泄漏自检 ===")
        leak_check(m, device=dev)
    except ImportError as e:
        print(f"  [跳过] 需要 snowflake_B.py 在同目录: {e}")
    except Exception as e:
        print(f"  [ERR] {type(e).__name__}: {e}")
