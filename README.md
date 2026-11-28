# Snowflake MoE / paperplaneMoe

一个从零实现的组合式 Mixture-of-Experts 研究仓库。当前主线是
`v2_compositional/`：先组合专家参数，再执行一次非线性前向，并在此基础上研究
可观测的“细胞/细胞器”类比和守恒的优化资源分配。

## 先读：项目的科学边界

本项目当前是**生物启发的机器学习模型**，不是完整细胞模拟器。代码里的
`bloodstream`、`mitochondria`、`chloroplast`、`enzyme` 等名称，部分只对应旁路
状态机或优化器类比；它们尚不构成真实代谢网络，也不能同时被解释为一种自然细胞。

- 逐模块审计与缺失环境变量：[`docs/BIO_CHEM_CELL_ENVIRONMENT_AUDIT.md`](docs/BIO_CHEM_CELL_ENVIRONMENT_AUDIT.md)
- 守恒更新层的数学和文献边界：[`docs/BIO_UPDATE_FOUNDATIONS.md`](docs/BIO_UPDATE_FOUNDATIONS.md)
- 本机性能测试和推荐配置：[`docs/PERFORMANCE_LOCAL.md`](docs/PERFORMANCE_LOCAL.md)

## 快速开始

安装依赖：

```bash
python -m pip install -r requirements.txt
```

Windows 本机 CPU 推荐入口（本机实测 2 个 intra-op 线程最快）：

```powershell
cd v2_compositional
.\run_local_cpu.ps1 -Python python -Data data_25mb.txt -Steps 200
```

等价的跨平台命令：

```bash
python v2_compositional/train_bio.py \
  --data v2_compositional/data_25mb.txt \
  --max-steps 200 --device cpu --cpu-threads 2 \
  --interop-threads 1 --token-cache auto
```

`--token-cache auto` 只缓存字符 token，不改变数据划分或随机采样。CUDA/XPU 必须
显式可用；指定了不可用设备时程序会报错，不会静默换成 CPU。

## 仓库地图

| 路径 | 状态 | 内容 |
|---|---|---|
| `v2_compositional/` | 当前研究主线 | 组合专家、V3/V4、BioUpdate、实验脚本 |
| `v2_compositional/paperplane_moe/` | 核心包 | 生物学类比原语、遥测、细胞环境契约 |
| `v2_compositional/tests/` | 自动验证 | 守恒数学、环境完整性、缓存等价性 |
| `docs/` | 文档中心 | 架构、协议、审计、负结果、性能 |
| 根目录 `run_*.py` | v1/历史实验 | 词级 CellMoE、擦除、终身学习、OOD |
| `README_v1_memory_points.md` | 历史定义 | v1 独立值槽“忆点” |
| `README_v2_main.md` | v2 概览 | v2 连接式“忆点”与结果摘要 |

完整文档索引见 [`docs/README.md`](docs/README.md)。旧文件暂不移动，避免破坏实验
脚本和结果 JSON 中已经记录的路径；整理采用“索引 + 明确版本边界”的方式。

## 可复现实验

V1.010 的 A 路线包含守恒基线、压力驱动和冻结对照三臂：

```bash
cd v2_compositional
bash run_step34_A.sh
python analyze_step34_A.py --outdir output/v4 --prefix st34 \
  --write output/v4/st34_A_summary.json
```

当前本地 5-seed 配对结果的 95% 区间跨 0，因此压力项相对基线应标记为
`inconclusive`，不能写成已证明提升。详见
[`v2_compositional/README.md`](v2_compositional/README.md)。

## 测试

```bash
python -m unittest discover -s v2_compositional/tests -p "test_*.py" -v
```

## 许可证与作者

Apache-2.0，见 [`LICENSE`](LICENSE)。作者：Wu Shangzhen（Independent Researcher）。
