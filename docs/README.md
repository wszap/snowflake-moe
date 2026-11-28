# 文档索引

本页只建立导航，不移动旧文件；这样历史命令、JSON 结果和论文草稿中的路径仍然有效。

## 当前入口

- [`../README.md`](../README.md)：仓库入口、版本边界、快速运行
- [`../v2_compositional/README.md`](../v2_compositional/README.md)：当前组合式 MoE 主线
- [`PERFORMANCE_LOCAL.md`](PERFORMANCE_LOCAL.md)：本机性能基准和设备配置

## 科学边界与审计

- [`BIO_CHEM_CELL_ENVIRONMENT_AUDIT.md`](BIO_CHEM_CELL_ENVIRONMENT_AUDIT.md)：逐模块生物/化学审计、细胞环境缺口
- [`BIO_UPDATE_FOUNDATIONS.md`](BIO_UPDATE_FOUNDATIONS.md)：守恒资源更新层的数学、文献和可证伪条件
- [`AUDIT_REPORT_1017.md`](AUDIT_REPORT_1017.md)：阶段性代码审计
- [`审核报告_v8_1018.md`](审核报告_v8_1018.md)：v8 中文审核记录
- [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md)：必须保留的负结果

## 架构

- [`ARCHITECTURE_V3.md`](ARCHITECTURE_V3.md)：V3 架构
- [`V4_VARIANTS.md`](V4_VARIANTS.md)：V4 变体
- [`COMPOSABLE_EXPERT_V8.md`](COMPOSABLE_EXPERT_V8.md)：组合专家定义
- [`CLONE_MECHANISM.md`](CLONE_MECHANISM.md)：复制机制
- [`KAKEYA.md`](KAKEYA.md)、[`KAKEYA_SEG.md`](KAKEYA_SEG.md)：相关几何探索

## 实验与复现

- [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md)：实验协议
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)：按时间记录的实验日志
- [`MoE六个失败模式.md`](MoE六个失败模式.md)：失败模式和反例
- [`blog_moe_six_traps.md`](blog_moe_six_traps.md)：面向读者的六陷阱说明

## 历史与写作

- [`idea_evolution.md`](idea_evolution.md)：想法演进
- [`blog.md`](blog.md)：项目长文
- [`CONTRIBUTING.md`](CONTRIBUTING.md)：贡献说明
- [`FLIPPER.md`](FLIPPER.md)：Flipper 相关记录

## 文档口径规则

1. v1 的独立值槽“忆点”和 v2 的连接式“忆点”不能混为同一机制。
2. 理论 FLOPs、wall-clock、显著性检验分别报告，不能互相替代。
3. 生物学名称默认是工程类比；只有通过环境、单位、守恒和动力学验证的部分，才可升级科学表述。
4. 95% 区间跨 0 的实验统一写 `inconclusive`，负结果不删除。
