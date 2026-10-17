# 换电脑迁移手册 — Snowflake MoE

> 目标：**不丢任何东西**，换机后能立刻继续工作。
> 三件事：① 本地归档推 GitHub ② 远端服务器取证 ③ 新电脑恢复。

---

## 总览

| 位置 | 内容 | 动作 |
|---|---|---|
| **本地沙盒** | 代码 + 文档（本归档包） | 推 GitHub backup 分支 |
| **远端服务器** | 10 seeds 证据、日志、ckpt、数据集 | `collect_remote_evidence.sh` 取证后 scp 下来 |
| **GitHub** | 仓库（v1.0 现状） | 推 backup 分支；整改走独立分支 |

---

## 步骤 1：推送本地归档到 GitHub（换机前，现在做）

```bash
# 1.1 克隆/进入仓库
git clone https://github.com/wszap/snowflake-moe.git
cd snowflake-moe

# 1.2 一键推送
export REPO=$(pwd)
bash <归档包>/scripts/push_all_to_github.sh
```

**默认推到 `backup/full-archive-MMDD` 分支，不动 main。**

推送内容：
- `v2_compositional/` —— 全部 v2.0 代码 + RESULTS.md
- `docs/` —— 博客、负结果、实验日志（37 条）
- `scripts/` —— 整改脚本、取证脚本、MARVIS 任务书
- `README_main.md` / `README_v1.md` / `MARVIS_TASK.md` / `INTEGRATION_PLAN.md`

---

## 步骤 2：远端服务器取证（换机前，现在做）

**⚠️ 这一步不能省** —— 10 seeds 的 `record.json` + `steps.csv` 只在服务器上。

```bash
# 在【服务器】上执行
bash collect_remote_evidence.sh
```

会生成 `~/snowflake_remote_backup_<时间戳>.tar.gz`，包含：

| 项 | 内容 |
|---|---|
| runs/ | 10 seeds 的 `record.json` + `steps.csv` |
| logs/ | 所有 `*.log` |
| ckpt/ | checkpoint（<100MB 的） |
| data_fingerprint/ | 数据集 md5 / 大小 / 字符数（**不复制数据本体**） |
| environment.txt | torch 版本、DCU 信息、磁盘 |
| code/ | 服务器上的代码快照 |

```bash
# 下载到本地
scp root@<服务器>:~/snowflake_remote_backup_*.tar.gz ~/
```

**数据集怎么办**：100MB 的 `tinystories_100mb.txt` 不进包（太大）。
换机后重新下载即可，或者用指纹校验是否同一份：

```bash
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz
tar -xzf TinyStories_all_data.tar.gz
md5sum tinystories_100mb.txt    # 与 data_fingerprint 对比
```

---

## 步骤 3：新电脑恢复

```bash
# 3.1 拉代码
git clone https://github.com/wszap/snowflake-moe.git
cd snowflake-moe
git checkout backup/full-archive-MMDD

# 3.2 解压远端证据
tar -xzf ~/snowflake_remote_backup_*.tar.gz
mv snowflake_remote_backup_*/runs  v2_compositional/output/runs
mv snowflake_remote_backup_*/logs  docs/remote_logs

# 3.3 装环境
pip install -r requirements.txt
```

---

## 步骤 4：让 MARVIS 执行仓库整改

按顺序执行（详见 `scripts/MARVIS_TASK.md`）：

```bash
git checkout -b restructure/v2-integration

# ① 环境检查（预期 5 条警告）
bash scripts/restructure_repo.sh --check

# ② 修 v1.0 文档硬伤（纯文本，零风险，可先单独推）
bash scripts/restructure_repo.sh --phase 1

# ③ 归档 v1.0 + 建目录（git mv，保留历史）
bash scripts/restructure_repo.sh --phase 2

# ④ 投放文件（按 MARVIS_TASK.md §4 的映射表）
cp scripts/README_main.md README.md
cp scripts/README_v1.md   v1_memory_points/README.md
# ... 其余见任务书

# ⑤ 跑 dense baseline（补 P0 证据）
cd v2_compositional
python dense_baseline.py --data <TinyStories> --scale-sweep
python dense_baseline.py --data <TinyStories> --L 5 --d 288 --sweep
python compare_dense.py --dense output/dense
```

---

## 五处硬伤（阶段 1 自动修）

| # | 问题 | 修法 |
|---|---|---|
| ① | `python run_lm.py` 文件不存在 | → `run_experiments.py` / `train_lm.py` |
| ② | OOD ratio<1 | 标题改 "OOD Robustness — Relative Degradation" |
| ③ | Parameter Efficiency 是 active/fixed | 改 "Sparse Activation Ratio"，注明非 vs dense |
| ④ | "a fraction of dense" 无数据 | 加「尚无 dense 对照」标注 |
| ⑤ | 忆点定义互斥 | 加 v1.0 值槽 / v2.0 连接本体版本标注 |

---

## 易混淆点（务必记住）

| 项 | v1.0 | v2.0 |
|---|---|---|
| 参数量 | **1,823,077** | **5,163,618** |
| 任务口径 | 词级 TinyStories | 字符级（vocab=98） |
| PPL | 10.0157 | ≈5.95（loss 1.788） |
| 忆点定义 | 独立值槽 | 连接本体 |

**两个版本的 PPL 不可比**，差近一倍量级。

**v2.0 参数量早期文档写「3.6M」是错的**，逐项核算是 5,163,618
（W1_U 384×128×4 = 196,608 是最大头）。

---

## dense baseline 关键配置（别改）

- 默认 `L=5, d=288, ff=1152` → 5,024,736（**0.973x** v2.0 的 5,163,618）
- **不加 pos embedding** —— SnowflakeLM 也没有；加了参数量虚高 9.9%，对比不公平
- 数据用 `--fix-mojibake`，与 Snowflake 侧一致

判定：
```
临界 dense 参数 / 5,163,618 < 1  → "a fraction" 成立
                            > 1  → 必须改 README
```

---

## 检查清单

**换机前**
- [ ] `push_all_to_github.sh` 已推送 backup 分支
- [ ] 服务器上 `collect_remote_evidence.sh` 已执行
- [ ] tar.gz 已 scp 到本地/网盘
- [ ] 数据集 md5 已记录（`data_fingerprint/`）

**换机后**
- [ ] `git checkout backup/full-archive-MMDD` 成功
- [ ] 远端证据已解压到 `v2_compositional/output/runs`
- [ ] `pip install -r requirements.txt` 完成
- [ ] `bash scripts/restructure_repo.sh --check` 能跑
- [ ] MARVIS 已按任务书整改

---

## 回滚

```bash
git tag                              # 找 pre-restructure-*
git reset --hard pre-restructure-XXXXXXXX
```
