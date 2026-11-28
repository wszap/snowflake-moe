# 生物学、化学与细胞环境审计

审计对象：`v2_compositional/paperplane_moe/biology.py`、`enzymology.py`、
`enzyme_rate.py`、`compose.py` 与 `train_bio.py`。

## 结论先行

当前代码是一个带生物学命名的**组合专家优化系统**，不是全细胞模型，也不是
满足化学计量和热力学约束的代谢模型。

- “拼专家”前向、关闭路径零扰动、局部随机数、可重放事件和守恒优化预算是
  可执行、可证伪的工程机制。
- `phi` 单纯形可作为有限蛋白资源分配的数学类比，但梯度 RMS 不是代谢通量、
  学习率不是酶速率常数、token 表征也不是物质浓度。
- 受体、血液、骨架、回顾、ATP、线粒体、叶绿体、自噬和凋亡大多是观测对象、
  建议事件或未接入训练的返回值。它们不能据此被称为已经发生的细胞过程。
- 当前没有明确的细胞外介质、细胞膜、体积、单位、离子、pH、渗透压、膜电势、
  氧化还原、废物和物质量守恒，因此“细胞环境”不完整。
- 同一个抽象体同时使用细菌的 lac-operon 式诱导、动物/组织层级的“血液”、
  真核线粒体和植物叶绿体。作为跨域灵感可以保留；作为一种自然细胞不相容。

仓库新增的 `paperplane_moe/cell_environment.py` 只建立显式单位与完整性契约，
暂不接入损失函数。缺少物理状态时宁可报告缺项，不用神经网络统计量伪装浓度。

## 代码实际作用面

| 名称 | 当前代码实际行为 | 生物学判定 |
|---|---|---|
| Receptor | 用 token 范数高于均值做标记；返回的扰动在训练中未使用 | 输入探针，不是膜受体/配体动力学 |
| Bloodstream | 保存向量并做 EMA；训练只对嵌入调用 `layer_idx=0`，返回值未进入模型 | 记忆总线类比，不是血流或主动运输 |
| Nucleoskeleton | 按 task id 索引低秩 buffer；训练传入零向量 | 状态缓存，不是细胞核骨架 |
| SparseRecall | 余弦 top-m 检索；返回值未进入模型 | 检索模块，不是生物回忆过程 |
| ATPBudget | 标量记账；训练只调用 `spend(0.0)` | 未与任何反应或运输耦合 |
| Enzyme (`biology.py`) | 对隐藏向量做中心化/标准化；返回值未使用 | 归一化类比，不是酶动力学 |
| Enzymology V1.003 | 维持无量纲 S、Km、浓度和 EMA；可乘专家输出幅值 | 具有动力学外形，但无化学计量/单位/平衡 |
| EnzymeRate V1.004 | 根据梯度统计修改 AdamW param-group 学习率 | 自适应优化器；不能称为真实酶速率常数 |
| Mitochondria | 梯度二阶矩预条件器；训练中没有调用 `precondition` | 未发生线粒体供能 |
| Chloroplast | 固定随机投影；训练返回值未使用 | 不是光反应或碳固定 |
| Autophagy | 只生成建议事件；训练固定传 `grad_starve=False` | 不执行物质回收 |
| Apoptosis | 只生成建议事件，不删除/冻结模型参数 | 不执行细胞死亡 |
| Division | 扩展专家张量和重建优化器 | 模型生长事件，不含 DNA/膜/细胞质分配 |
| BioUpdate | 在参数组间守恒搬运学习率预算 | 数学守恒成立；属于蛋白资源分配类比 |

## 细胞环境覆盖矩阵

“部分”表示有同名观测量或类比，但没有对应物理状态方程。

| 必要轴 | 当前状态 | 主要缺口 |
|---|---|---|
| 边界与几何 | 缺失 | 无膜面积、体积、区室拓扑 |
| 温度 | 缺失 | 无 K、Arrhenius/Eyring 温度依赖 |
| 内外 pH | 缺失 | 无质子、缓冲容量、pH 稳态 |
| 离子组成 | 缺失 | 无 Na/K/Cl/Ca/Mg 与电荷平衡 |
| 渗透状态 | 缺失 | 无水、渗透压或体积调节 |
| 膜电势 | 缺失 | 无电容、离子通量和质子动力势 |
| 营养源 | 部分 | token 激活量被叫作“底物”，没有物质身份 |
| 电子受体 | 缺失 | 无 O2/替代受体和电子传递链 |
| 氧化还原 | 缺失 | 无 NAD(H)、NADP(H)、FAD 或氧化还原势 |
| 能量货币 | 部分 | ATP 只有标量预算，不含 ATP/ADP/Pi 守恒 |
| 废物/产物 | 部分 | `product_ema` 是速率 EMA，不是产物浓度 |
| 信号 | 部分 | 受体标记无配体、亲和力和占据率 |
| 力学 | 缺失 | 无压力、张力、黏度、拥挤或形变 |
| 空间/扩散 | 缺失 | 所有状态默认瞬时全混合 |
| 时间尺度 | 部分 | step/dt 无物理单位，各过程无多时间尺度校准 |
| 随机性 | 部分 | 有局部 PRNG，但不是化学主方程/Gillespie 噪声 |

## 化学与物理硬约束

### 1. 质量与元素守恒

真实反应网络至少需要化学计量矩阵 `N`、物质量 `n` 和通量 `v`，使封闭部分满足
`dn/dt = N v`。当前 token 激活、梯度和学习率没有元素组成，因此不能验证 C、N、P、
电荷或总质量守恒。`sum(phi)=1` 只证明优化预算守恒。

### 2. 热力学方向性

反应方向应与化学势/吉布斯自由能相容；可逆酶参数还应与平衡常数保持 Haldane
一致性。当前没有正逆反应、活度、标准生成能或 `ΔG`，所以不能声称详细平衡、
Haldane 关系或“酶不改变平衡”已由代码实现。

### 3. 酶动力学

V1.003 的饱和函数与诱导/降解慢变量适合作为控制器原型，但参数无单位，底物是
前向激活统计，`product_ema` 也不是由化学计量产生的产物。固定 Hill 系数 2 不等于
MWC 的多亚基构象平衡。V1.004 修改学习率后，科学名称应是“酶启发的自适应更新率”。

### 4. 跨膜运输和供能

主动运输必须与膜、浓度梯度、电化学势及 ATP/质子耦合。当前 `active_transport`
只改变 telemetry 标签；线粒体也未建立呼吸链、质子动力势或 ATP 合成。叶绿体的
随机矩阵没有光子吸收、电子传递或碳固定。

## 可保留的科学价值

1. `BioUpdate` 的资源份额位于单纯形，参数量加权学习率预算每步守恒；这是清楚的
   约束优化模型，可与蛋白资源分配作有限类比。
2. 关闭路径、冻结对照和分裂重同步都有硬闸，优于只凭故事解释模型。
3. 事件记录、谱系和多 seed 配对分析提供了可证伪的实验框架。
4. “拼参数再做一次非线性前向”是明确的模型架构事实，不依赖生物学命名成立。

## 后续接线顺序

1. **先选物种/层级**：细菌、动物细胞、植物细胞或明确写成 abstract mixed-cell。
2. **只建立环境状态**：使用 `CellEnvironment` 填入单位、体积和内外化学池；未通过
   `audit()` 不准接入训练。
3. **再建立运输账本**：每个跨膜事件同时更新内外物质量、电荷和 ATP/离子耦合。
4. **再建立反应网络**：给出化学计量、可逆性、速率律、参数单位和热力学方向。
5. **最后映射到优化器**：只有在干预实验表明环境状态改变模型行为后，才把它接入
   学习率、路由或专家生长；必须保留无环境、冻结环境和预算匹配对照。

## 一手资料与定量参照

1. Karr et al. (2012), *A Whole-Cell Computational Model Predicts Phenotype from Genotype*.
   DOI: https://doi.org/10.1016/j.cell.2012.05.044
2. Scott et al. (2010), *Interdependence of Cell Growth and Gene Expression*.
   DOI: https://doi.org/10.1126/science.1192588
3. Jacob & Monod (1961), *Genetic Regulatory Mechanisms in the Synthesis of Proteins*.
   DOI: https://doi.org/10.1016/S0022-2836(61)80072-7
4. Monod, Wyman & Changeux (1965), *On the Nature of Allosteric Transitions*.
   DOI: https://doi.org/10.1016/S0022-2836(65)80285-6
5. Mitchell (1961), *Coupling of Phosphorylation to Electron and Hydrogen Transfer by a
   Chemi-Osmotic Type of Mechanism*. https://www.nature.com/articles/191144a0
6. Kacser & Burns (1973), *The Control of Flux*.
   https://health.uconn.edu/quantitative-medicine/wp-content/uploads/sites/117/2016/08/KacserBurns1973_21yearsOn_Pedro.pdf
7. Bennett et al. / quantitative metabolome constraint work summarized with explicit
   thermodynamic, charge, membrane-potential, buffering, ionic-strength and osmotic constraints:
   https://pmc.ncbi.nlm.nih.gov/articles/PMC8155068/
