---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 39b3c1e52d56df687e379d81ca7e9c3c_55af0391c07f11f1a05452540064ee0f
    ReservedCode1: auz84bO59M1W+yI546A83rlVgeXoN/rZuJgA+tNd4bMNTtCkd0PiwsIuwnUnRZW5fIqBHQZezcd8p0PzN1Xsc6rxMVupA/qGSr0epyQYq11zSu9DCt70k44y2eVcepi/Tga+RAk8fou6WM0U5Zi8J28dcq+FWY1LCkvDreOnratMHSZPA5r1jtqI5/s=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 39b3c1e52d56df687e379d81ca7e9c3c_55af0391c07f11f1a05452540064ee0f
    ReservedCode2: auz84bO59M1W+yI546A83rlVgeXoN/rZuJgA+tNd4bMNTtCkd0PiwsIuwnUnRZW5fIqBHQZezcd8p0PzN1Xsc6rxMVupA/qGSr0epyQYq11zSu9DCt70k44y2eVcepi/Tga+RAk8fou6WM0U5Zi8J28dcq+FWY1LCkvDreOnratMHSZPA5r1jtqI5/s=
---

# Snowflake MoE：从零实现一个可擦除、可终身学习的稀疏 MoE

> 作者：snowflakewsz
> 日期：2026-10-05

## 为什么做这件事

大模型的参数越来越多，但大部分参数在每次前向时都是"沉睡"的。稀疏 MoE（Mixture-of-Experts）的思路是：把网络拆成多个专家，每次只激活其中一小部分，用远少于稠密模型的活跃参数达到同等能力。

但标准 MoE 有几个老问题：路由坍缩（少数专家被反复选中）、负载不均衡、知识高度耦合（想删掉某个领域的知识只能整体重训）。这个项目从零实现了自己的答案——**Snowflake MoE**：细粒度专家 + 共享专家 + 稀疏激活的组合式路由，以及可定位、可擦除的记忆槽机制。

## 架构要点

- **组合式 MoE**：路由器学习把细粒度专家组合成可复用的"专家子集"，在极小参数预算下获得强泛化。
- **记忆点（Memory Points）**：事实知识以解耦的记忆值形式存储在专用槽位中，与路由权重分离。
- **可擦除记忆**：推理时可选择性归零记忆槽。擦除 50% 槽位只带来轻微 PPL 退化，擦除 100% 则模型显著劣化——证明知识确实被"局部化"存放，且可按需擦除。
- **终身学习**：顺序加入新领域（TinyStories → 圣经语料）时，旧领域几乎无退化。

## 关键实验数据

### 1. 基线能力（TinyStories 字符级续写）

| 模型 | 参数 | val PPL |
|---|---|---|
| CellMoE | 1.82M | 10.0157 |
| Fixed MoE（对照） | 1.89M | 10.0152 |

在 ~200 万参数规模下达到 PPL ≈ 10，且稀疏模型的活跃参数远少于固定 FFN 对照。

### 2. 可擦除性（证明知识局部化）

| 擦除比例 | val PPL（CellMoE） | val PPL（MR2 变体） |
|---|---|---|
| 0% | 10.0148 | 10.0169 |
| 50% | 10.1261（+1.1%） | 11.2788（+12.6%） |
| 100% | 18.1251（+81%） | 24.0217（+140%） |

擦除 100% 记忆值后模型明显劣化，说明关键知识确实存储在记忆槽中；擦除 50% 影响温和，说明知识分布有一定冗余。

### 3. 终身学习（旧域几乎无退化）

TinyStories 学完后加入圣经语料：

| 阶段 | TinyStories PPL | 圣经 PPL | 新记忆利用率 |
|---|---|---|---|
| 基线 | 10.0148 | 19.41 | 0% |
| 终身学习初版 | 10.0246（+0.098） | 18.22（-6.1%） | 0%（未达标） |
| v4（InputAwareGate） | 16.68 | 16.10（-21.64%） | 100% |

初版暴露了关键问题：新增记忆点利用率 0%，即门控没有真正使用新记忆。v4 引入输入相关门控（InputAwareGate）后，新记忆利用率达到 100%，新域 PPL 下降 21.64%，超过 20% 验收线。

### 4. 速度剖析（纯计算 vs 环境噪声）

控制变量重跑（剥离温度暂停等环境噪声）后测得：CellMoE 的纯计算耗时是 Fixed FFN 对照的 **1.75×**。这个差距来自动态路由、多专家前向与记忆读取，是后续优化的重点对象。

## 从零到超算：一次完整的工程闭环

项目在本地 RTX 5060 Laptop（8GB）上完成了机制验证，随后部署到曙光超算 DCU（64GB 显存，PyTorch 2.9.0）：

- 超算外网不通，依赖全部通过本地下载 Linux wheel 离线安装；
- numpy 必须锁 1.x（torch 2.9.0 编译于 numpy 1.x，numpy 2.x 会初始化失败）；
- DCU 环境需要先 `source /opt/dtk-26.04/env.sh` 才能加载运行时；
- 通过 `--data` 参数直接加载预处理好的 `.pt` 数据，绕开在线下载。

## 教训与红线

1. 加载 checkpoint 时缩放参数必须与训练一致（scale mismatch 会导致灾难性遗忘）。
2. 规模化扩展（100B 级）需要专家并行、All-to-All 通信、BF16/FP8、激活检查点、分布式检查点等一整套工程手段，不能只靠单机脚本。
3. 评价一个机制改进，必须用控制变量剥离环境噪声后再下结论。

## 开源

代码已整理并开源：`opensource` 目录包含稳定核心（`marvis_moe.py` / `snowflake_moe*.py` / `train_snowflake.py`）、复现命令与完整实验归档（`early_experiments/`）。checkpoints、数据、日志不纳入仓库。
*（内容由AI生成，仅供参考）*
