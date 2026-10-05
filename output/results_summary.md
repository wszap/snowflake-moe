---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 39b3c1e52d56df687e379d81ca7e9c3c_56b6ce91c07f11f1bc7f525400638852
    ReservedCode1: t87afQ8BBxbJrg+4N+YBWwSHIDG1PVmv/TyhEOMxngzrNO51OWT6fYbw24zZhhGq/E5dQbbZNaS2Jo+o4kuJbPHgYgmGsgr3Oa5xdQ6BcBbdwKjOoNCcXeb4Z/KMWQ8njqY4iWvvp8gUCXqWOLdopnVsUWWDXxNwfPcNwZzVMhb68VSdl0dSOP++EDY=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 39b3c1e52d56df687e379d81ca7e9c3c_56b6ce91c07f11f1bc7f525400638852
    ReservedCode2: t87afQ8BBxbJrg+4N+YBWwSHIDG1PVmv/TyhEOMxngzrNO51OWT6fYbw24zZhhGq/E5dQbbZNaS2Jo+o4kuJbPHgYgmGsgr3Oa5xdQ6BcBbdwKjOoNCcXeb4Z/KMWQ8njqY4iWvvp8gUCXqWOLdopnVsUWWDXxNwfPcNwZzVMhb68VSdl0dSOP++EDY=
---

# Snowflake MoE 实验结果汇总（截至 2026-10-05）

> 由 Marvis 汇总本地 `output/*.csv` 生成，供报告与博客引用。

## 1. TinyStories 基线（val PPL）

| 模型 | 参数 | val_ce | val_ppl | 耗时(sec) | 备注 |
|---|---|---|---|---|---|
| CellMoE (ckpt) | 1,823,077 | 2.3041 | 10.0148 | 1376.9 | 首个基线 |
| CellMoE | 1,823,077 | 2.3042 | 10.0157 | 2276.6 | 补跑 |
| CellMoE (MR2) | 1,823,077 | 2.3043 | 10.0169 | 1557.0 | memory_read×2.0 |
| Fixed FFN E14 | 1,886,053 | 2.3041 | 10.0152 | 1266.1 | 对照 |

结论：CellMoE 与 Fixed 对照 PPL 持平（~10.01），参数量相近，但 CellMoE 活跃参数更少。

## 2. 可擦除性（TinyStories）

| 模型 | 擦除 0% | 擦除 50% | 擦除 100% |
|---|---|---|---|
| CellMoE | 10.0148 | 10.1261 (+1.1%) | 18.1251 (+81%) |
| CellMoE MR2 | 10.0169 | 11.2788 (+12.6%) | 24.0217 (+140%) |

结论：知识局部化于记忆槽；100% 擦除显著劣化，50% 温和退化。

## 3. 终身学习（TinyStories → Bible）

| 阶段 | ts_base_ppl | ts_life_ppl | ts_regress | bible_base_ppl | bible_life_ppl | bible_drop | new_mem_util | pass |
|---|---|---|---|---|---|---|---|---|
| 初版 group_softmax | 10.0148 | 10.0246 | 0.0098 | 19.4129 | 18.22 | 6.14% | 0.0% | FAIL |
| v4 InputAwareGate | 16.6772 | 16.1025 | - | - | - | 21.64% | 100% | FAIL(scale) |

结论：初版新记忆利用率 0%（门控未使用新记忆）；v4 达 100% 利用率、Bible PPL 降 21.64%，但旧域退化超 0.3 验收线（scale mismatch 导致），需修正后复验。

## 4. 速度（控制变量，TinyStories 5 epoch）

| 模型 | total_sec | pure_compute_sec | gpu_pauses | 相对 Fixed |
|---|---|---|---|---|
| CellMoE_v2 | 2216.3 | 2216.3 | 0 | 1.75× |
| Fixed（历史 log） | 1266.1 | - | - | 1.00× |

结论：真实计算慢 1.75×，非环境噪声；列入优化清单（目标 1.15–1.3×）。

## 5. 莎士比亚小模型（snowflake_v5）

| 指标 | 值 |
|---|---|
| val_ppl | 14.1196 |
| mem_util | 1.0 |
| org_util | 0.7744 |
| params | 207,033 |
| fixed_params | 259,009 |
| param_ratio | 0.799 |

## 6. 莎士比亚擦除/终身学习（早期原型）

| 阶段 | shakespeare_ppl | 恢复性 |
|---|---|---|
| baseline | 12.8126 | - |
| erased_50 | 12.8174 | - |
| erased_100 | 13.7804 | - |
| recover_50 | 12.3148 | yes |
| recover_100 | 12.3697 | yes |

终身学习：lifelong_1epoch 新域 15.27 → 14.16，new_memory_utilization=0.34。

## 7. OOD 泛化

| 模型 | id_ppl | ood_ppl | ood_id_ratio |
|---|---|---|---|
| cellmoe_v3 | 15.0317 | 13.9253 | 0.9264 |
| fixed_moe | 26.5113 | 18.1065 | 0.683 |

结论：CellMoE 组合式路由在 OOD 上泛化优于 Fixed 对照。
*（内容由AI生成，仅供参考）*
