# V1.010 守恒更新层：科学依据、实现和边界

## 1. 它仍然是“拼专家”

本次改动不碰 `paperplane_moe/compose.py` 的核心前向：路由权重先组合器官参数，随后只进行一次非线性前向。它没有改成逐 token 选择若干完整专家，也没有新增 top-k 路由。

更新层只处理训练资源如何在 `shared + 每个 compositional cell` 之间分配。模型前向和优化器资源分配是两个正交层次。

## 2. 资源变量

设优化器共有 `G` 个区室。第 `i` 组的参数量占比为：

```text
φ0_i = n_i / Σ_j n_j
```

`φ0` 是零扰动基线；所有组的学习率都等于 scheduler 给出的 `η_base`。

动态资源份额 `φ` 始终位于单纯形：

```text
φ_i > 0,  Σ_i φ_i = 1
```

相对学习率和实际学习率为：

```text
s_i = φ_i / φ0_i
η_i = η_base · s_i
Σ_i φ0_i · s_i = 1
```

最后一个恒等式是代码真正保证的“预算守恒”：它是参数量加权的平均学习率守恒，不是把每组当成同样大小后简单相加。

## 3. 数学更新

使用指数梯度/负熵镜像下降：

```text
φ' ∝ exp(log φ + ηφ [wC·q − κ·log(φ/φ0)])
```

其中 `q` 是每组梯度 RMS 慢 EMA 的中心化对数比。它只叫“优化压力代理”，不冒充实验测得的代谢通量控制系数。`wC=0` 时，`φ=φ0` 是严格不动点；`wC>0` 是需要通过多 seed 实验验证的处理臂。

`bio_trust` 通过 capped-simplex 水位投影实现 `φ_i ≤ trust·φ0_i`，投影后仍保持 `Σφ=1`。`trust=1` 的唯一可行点就是 `φ0`，因此是机制冻结对照。

## 4. 生物学对应关系

- 蛋白质组资源分配：细胞总蛋白资源受约束，支持把 `φ` 解释为有限资源份额的建模同构。Scott 等人的实验与理论工作给出了细菌生长、翻译资源和无收益蛋白表达之间的定量约束（Science 2010，DOI `10.1126/science.1192588`）。
- 代谢控制分析：Kacser–Burns 的通量控制系数满足加和关系，但它定义于稳态代谢网络的局部灵敏度。本项目没有直接测量该量，所以梯度 RMS 不命名为 `C^J_i`。
- 变构：Monod–Wyman–Changeux 模型要求寡聚体的协同构象转换。一个对所有参数组相同的标量在 softmax 中会完全抵消，因此 V1.010 不再把全局 `R/T` 标量伪装成组间分化信号。

## 5. 化学与物理边界

- Haldane 关系约束可逆酶反应的动力学参数与平衡常数；AdamW 没有可识别的底物、产物、正反反应速率，因此本更新层不宣称满足 Haldane 关系。
- 随机热力学的局部详细平衡约束正反跃迁率比；当前离散优化器不是经过热浴标定的马尔可夫动力学，因此不把“详细平衡”写成测试结论。
- `loss` 下降不是化学产物浓度，step 也不是物理时间。它们只能作为遥测，不能进入带单位的化学公式。

## 6. 实现硬闸

每一步写入 `<tag>_bio.jsonl`：

- `G` 与稳定组名；
- `φ / φ0 / scale`；
- `Σφ` 与参数量加权 `budget_gap`；
- `scale_std`（作用面强度）；
- `resync`（split 重建优化器后的重新分组证据）。

以下任一情况返回失败：

- `G < 2`；
- `|Σφ−1| > 1e-9`；
- `|budget_gap| > 1e-9`；
- 处理臂 `max(scale_std) < 0.005`；
- 发生 split 的试验没有 `resync=true`。

## 7. 一手资料

1. Kivinen, J. & Warmuth, M. K. (1997). *Exponentiated Gradient versus Gradient Descent for Linear Predictors*. Information and Computation 132, 1–63. https://mwarmuth.bitbucket.io/pubs/J36.pdf
2. Monod, J., Wyman, J. & Changeux, J.-P. (1965). *On the Nature of Allosteric Transitions*. J. Mol. Biol. 12, 88–118. DOI: https://doi.org/10.1016/S0022-2836(65)80285-6
3. Kacser, H. & Burns, J. A. (1973; reprinted 1995). *The Control of Flux*. Biochemical Society Transactions. https://health.uconn.edu/quantitative-medicine/wp-content/uploads/sites/117/2016/08/KacserBurns1973_21yearsOn_Pedro.pdf
4. Scott, M. et al. (2010). *Interdependence of Cell Growth and Gene Expression*. Science 330, 1099–1102. DOI: https://doi.org/10.1126/science.1192588
5. Seifert, U. (2012). *Stochastic thermodynamics, fluctuation theorems and molecular machines*. Rep. Prog. Phys. 75, 126001. DOI: https://doi.org/10.1088/0034-4885/75/12/126001
6. Puigbò, P. et al. (2010). *Integrated stoichiometric, thermodynamic and kinetic modelling of steady state metabolism*. BMC Systems Biology 4, 49. https://pmc.ncbi.nlm.nih.gov/articles/PMC2868105/
7. Puigcerver, J. et al. (2023). *From Sparse to Soft Mixtures of Experts*. https://arxiv.org/abs/2308.00951
8. Wu, X. et al. (2024). *Mixture of LoRA Experts*. ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/file/ce806d8b4bf38cd92d483e5a0490d983-Paper-Conference.pdf

第 7–8 项只用于区分软组合、参数组合和传统选择式 MoE；paperplaneMoe 的具体“先组合参数、再过一次非线性”仍以仓库实现和消融结果为准。
