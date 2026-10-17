# 运行速查（修好后完整版）

## 0. 传包到服务器

```bash
scp snowflake_open.zip root@<server>:/root/private_data/snowflake/
# 远端
cd /root/private_data/snowflake && unzip -o snowflake_open.zip && cd snowflake_open
```

## 1. 摸底（3000 步）—— 先确认能跑通

```bash
python train_lock60.py --scale tiny \
    --data /root/private_data/snowflake/tinystories_100mb.txt \
    --delta-scale 0.5 --min-eff 8 --no-bf16 \
    --max-steps 3000 --seed 0 \
    --save-ckpt /root/private_data/snowflake/ckpt_mineff8.pt
```

看四件事：
- `[device] GPU OK BW ...`  ← 确认 DCU 在用
- `[data] ... vocab=100`    ← 确认字符级接入
- `loss` 从 4.60（=ln100）下降
- **`eff=`** ← 核心：每样本真正用几个器官

## 2. ★ release test —— 判定"真想拼"还是"被摁着"

这是判断拼专家成立与否的**唯一**实验。

```bash
python train_lock60.py --scale tiny \
    --data /root/private_data/snowflake/tinystories_100mb.txt \
    --delta-scale 0.5 --min-eff 8 --no-bf16 \
    --max-steps 3000 --release-steps 200 --seed 0 \
    --save-ckpt /root/private_data/snowflake/ckpt_rel.pt
```

输出：
```
  约束开启: sharp=0.xxxx  eff=8.xx
  约束关闭: sharp=0.xxxx  eff=?.??
  回弹幅度: XX.X%
  判定: OK 真想拼 / WARN 部分内化 / FAIL 被摁着
```

| 回弹 | 判定 | 论文怎么写 |
|---|---|---|
| < 25% | 真想拼 | 可写"自发学会组合" |
| 25~60% | 部分内化 | 需更长训练或降 K |
| > 60% | 被摁着 | 只能写"架构强制" |

## 3. ablation 崩了？从 ckpt 恢复评估，不用重训

```bash
python train_lock60.py --scale tiny --no-bf16 \
    --eval-only /root/private_data/snowflake/ckpt_mineff8.pt
```

## 4. 对照：同样步数跑 --min-eff 0

量化"强制组合"的代价。若 loss 差不多 ⇒ 白赚的组合度，强结果。

```bash
python train_lock60.py --scale tiny \
    --data /root/private_data/snowflake/tinystories_100mb.txt \
    --delta-scale 0.5 --min-eff 0 --no-bf16 --max-steps 3000 --seed 0
```

## 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--scale` | — | tiny / mid / large / extreme |
| `--data` | — | txt 路径 |
| `--min-eff` | 0 | 最低有效器官数。**不能开太大**（K=32 ⇒ 83% 均匀） |
| `--temp` | 5.0 | ⚠ N=384 下 2.0 会过均匀 |
| `--rank` | 4 | ⚠ B 型不能开大：r=8→1.078x |
| `--no-bf16` | 关 | **DCU 上必加**（实测 bf16 慢 2 倍） |
| `--release-steps` | 0 | release test 步数，建议 200 |
| `--max-steps` | 0 | 摸底用 3000 |
| `--save-ckpt` / `--eval-only` | — | 崩了可恢复评估 |
| `--hinge-low` | 关 | cos 向下漂移时开 |
| `--seed` | 0 | 多 seed 用 |

## 关键判据

| 指标 | 目标 | 含义 |
|---|---|---|
| `eff_organs` | ≥ 8 | 每样本真正用的器官数 ← 核心 |
| `in_band` | ≥ 0.7 | 连接健康 |
| `mix_sharp` | ≤ 0.7 | 非 one-hot |
| loss（字符级 3000 步） | 1.8~2.2 | >2.5 有问题 |
| loss（完整 epoch） | 1.3~1.6 | 健康 |

## 证据落盘

```
output/runs/<run_id>/record.json   含 git/config/数据指纹/verdicts
output/runs/<run_id>/steps.csv     每 100 步时序（含 eff）
output/runs/master.csv             横向汇总
```

⚠ 缺 `data_sha256` 的 run 会被标 `meta_missing`，不能进论文主表。
