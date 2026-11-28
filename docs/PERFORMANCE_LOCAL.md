# 本机性能基准（2026-11-28）

## 测试环境

- CPU：Intel Core Ultra 7 155H
- 内存：32 GB
- GPU：Intel Arc 集成显卡
- 当前显卡驱动：`32.0.101.8508`
- 当前项目 PyTorch：`2.15.0+cpu`
- 当前状态：CPU 可用；当前 wheel 未编译 XPU 后端

Intel 的 Windows PyTorch 前置条件列出 Meteor Lake-H/Intel Arc 支持，但要求的
WHQL 驱动为 `32.0.101.8801` 或更高；本机当前驱动低于该版本。因此仓库已加入
`--device xpu` 代码路径，但本次没有把 XPU 性能写成“已验证”。升级驱动和安装
对应 XPU wheel 属于系统级变更，应由用户确认后单独执行。

- [Intel PyTorch prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-13.html)
- [PyTorch XPU documentation](https://docs.pytorch.org/docs/stable/xpu.html)

## CPU 线程扫描

同一紧凑残差模型、同一 seed、100 步；最终验证损失均为约 `4.3573207`。

| intra-op 线程 | 总用时（秒） | ms/step |
|---:|---:|---:|
| 1 | 1.5796 | 15.7959 |
| **2** | **1.4955** | **14.9548** |
| 4 | 1.5601 | 15.6013 |
| 8 | 1.8723 | 18.7232 |
| 16 | 2.0664 | 20.6637 |

该小模型以 2 线程最佳；16 线程耗时约为 2 线程的 `1.38×`。原因是算子较小，
线程调度和同步成本超过并行收益。这个结论不应直接外推到更大模型，模型规模改变后
需要重新扫描。

## 数据缓存

25,300,766 字节语料、25,265,019 个字符 token、词表 101：

| 加载路径 | 时间（秒） |
|---|---:|
| 首次解码、建表并写缓存 | 2.1596 |
| 缓存命中 | 0.0370 |

缓存命中约 `58×`，缓存文件约 25.27 MB。同一 seed 下首批 `x/y` 张量逐位相等。
缓存指纹包含源文件绝对路径、大小、纳秒 mtime、乱码修复选项和格式版本；源数据变化后会重建。
缓存采用临时文件后原子替换，`*.tokens.pt` 已被 Git 忽略。

## 推荐命令

```powershell
cd v2_compositional
.\run_local_cpu.ps1 -Python python -Data data_25mb.txt -Steps 200 -Threads 2
```

或直接运行：

```bash
python train_bio.py --data data_25mb.txt --max-steps 200 \
  --device cpu --cpu-threads 2 --interop-threads 1 --token-cache auto
```

关键参数：

- `--device auto|cpu|cuda|xpu`：明确设备；显式设备不可用时立即失败。
- `--cpu-threads N`：只在 CPU 模式限制 intra-op 线程。
- `--interop-threads N`：限制算子间并发；本机入口使用 1。
- `--token-cache auto`：使用 `<data>.tokens.pt`。
- `--eval-batches N`：短程 smoke 可减小，正式实验保持默认 20。

## 仍可继续优化的层次

1. 在更新驱动和正确 XPU wheel 后，先做逐位/容差等价检查，再比较 XPU wall-clock。
2. 用 PyTorch profiler 定位组合权重构造、Python 循环和小矩阵 kernel 的真实占比。
3. 只对稳定形状测试 `torch.compile`；编译预热必须从正式计时中分离。
4. 批量化当前逐 cell/逐 organelle 的 Python 循环；这是比盲目增加线程更可靠的方向。
5. 对大模型重新扫描线程数、batch size 和序列长度，不固定沿用 2 线程结论。
