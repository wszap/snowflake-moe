# A10 服务器执行全流程

> 前置：本地已推送 `backup/full-archive-1017` 分支（含 `audit_ci.py` / `patch_all.py` / `run_after_patch.sh`）
> 工作目录：`/mnt/workspace/snowflake-moe`

---

## 第 0 步：确认环境

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
cd /mnt/workspace/snowflake-moe && git branch --show-current
ls -la v2_compositional/data_25mb.txt
```

判据：
```
NVIDIA A10                      ✅
backup/full-archive-1017        ✅
data_25mb.txt ≈ 25,035,796 字节 ✅
```

---

## 第 1 步：拉取最新代码

```bash
cd /mnt/workspace/snowflake-moe
git fetch origin
git checkout backup/full-archive-1017
git pull origin backup/full-archive-1017

# 确认新脚本到位
ls -la scripts/audit_ci.py scripts/patch_all.py scripts/run_after_patch.sh
```

三个文件都要存在。**缺任一个说明本地推送没成功，回去补推。**

---

## 第 2 步：脚本就位

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
cp ../scripts/run_after_patch.sh . 2>/dev/null
cp ../scripts/audit_ci.py .        2>/dev/null
cp ../scripts/patch_all.py .       2>/dev/null
chmod +x run_after_patch.sh
ls -la run_after_patch.sh audit_ci.py patch_all.py
```

---

## 第 3 步：存档 + 检查

```bash
bash run_after_patch.sh check
```

会做：确认 10 seeds 已完成 → 把现有 `output/runs/` 打包存档（避免补丁后混淆）→ 检查待修 bug。

**存档路径记一下**，形如 `output/runs_archive_<时间戳>/`。

---

## 第 4 步：★ CI 审计（10 秒，不占 GPU）

```bash
bash run_after_patch.sh audit
```

用**已有的 10 seeds 数据**重算统计，验证 Bug9 修正后结论是否稳固。

### 预期输出

```
core_stat = 0.7663 ± 0.0345 (n=10 seeds)
95% CI    = [0.7416, 0.7910]
t(9)      = 70.3     (临界 2.262)
修正后 CI 宽度 ≈ 0.0146（旧算法 0.0032，放大 4.6 倍）
10/10 pass  →  H4 依然成立
```

**只要看到 `10/10 pass`，论文结论就稳固了** —— 这一步不用等 75 分钟。

若出现 `fail`，把完整输出发我，别往下走。

---

## 第 5 步：打 9 个补丁

```bash
bash run_after_patch.sh patch
```

自动备份原文件（`.bak_patch<时间戳>`），修复：

| # | 问题 |
|---|---|
| Bug1 | `key_map` 缺 `eff_organs` → eff 从未落盘 |
| Bug2 | `finalize` 缺 eff/variance 映射 |
| Bug3 | H2 阈值 0.05 数学上不可达（N=384 上限 0.0026） |
| Bug4 | 采样每 100 步 → 早期剧变全漏 |
| Bug5 | sharp 告警计数采样遗漏（step 10~90 全错过） |
| Bug6 | H2 键名 `variance` vs `var_norm` 不匹配 |
| Bug7 | 模型侧无逐步计数器 |
| Bug8 | `dense_baseline` 默认 lr 3e-3，与 Snowflake 3e-4 差 10 倍 |
| **Bug9** | **CI/p 值用 `n_perm*n_batch=1000`，伪重复且非配对** |
| Bug10 | Bug9/5 新增字段未注册 → finalize 抛错 |

验证：
```bash
bash run_after_patch.sh verify
python3 -c "import ast;[ast.parse(open(f).read()) for f in ['run_metrics.py','train_lock60.py','snowflake_B.py']];print('语法 OK')"
```

---

## 第 6 步：重跑 10 seeds（约 75 分钟）

```bash
bash run_after_patch.sh rerun
```

这次 **eff / wiring_var_norm 会真正落盘**，H2 也能被评估。

### 监控（另开终端，Ctrl+C 不中断训练）

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
tail -f seeds.log
grep -c "═══ SEED" seeds.log      # 进度
grep -q ALL_DONE seeds.log && echo "完成" || echo "进行中"
```

### 跑完提取

```bash
python3 -c "
import json,glob,os,math,statistics as st
ds=sorted(glob.glob('output/runs/*/'),key=os.path.getmtime)[-10:]
cs=[];effs=[];vns=[]
print(f\"{'run_id':24} {'core_stat':>10} {'eff':>7} {'var_norm':>9}\")
for d in ds:
    r=json.load(open(d+'record.json')); f=r['final']
    c=f.get('core_stat'); e=f.get('eff_organs_final'); vn=f.get('wiring_var_norm')
    cs.append(c); effs.append(e if e else float('nan')); vns.append(vn if vn else float('nan'))
    print(f\"{d.split('/')[-2]:24} {c:>10.4f} {str(round(e,2) if e else '-'):>7} {str(round(vn,4) if vn else '-'):>9}\")
m=st.mean(cs); sd=st.stdev(cs)
print()
print(f'core_stat = {m:.4f} ± {sd:.4f}   CV={sd/m*100:.1f}%')
print(f't(9) = {m/(sd/len(cs)**0.5):.1f}   全部>0 = {all(c>0 for c in cs)}')
print(f'eff   = {st.mean(effs):.2f} ± {st.stdev(effs):.2f}')
print(f'var_norm = {st.mean(vns):.4f}   (H2 阈值 0.05)')
"
```

### 判据

| 指标 | 期望 | 说明 |
|---|---|---|
| core_stat | 0.766 ± 0.035 | 应与补丁前一致（Bug9 不改点估计） |
| **eff** | **21~27** | 补丁后终于落盘 |
| **var_norm** | 实测 ≈0.027 | ⚠️ 可能低于 0.05 |

---

## 第 7 步：dense 对比实验

```bash
bash run_after_patch.sh dense
```

内部顺序：
1. `--leak-check-only` —— 必须 `✅ 无泄漏`
2. 规模扫描（6 档，约 18 分钟）
3. 临界配置 10 seeds（约 30 分钟）
4. `compare_dense.py` 自动判定

### ⚠️ 泄漏自检必须过

causal mask 之前写反过（`triu` + SDPA 的 `True=参与` ⇒ 模型只看未来，loss 崩到 0.019）。

```
✅ 无泄漏（正确 causal）  → 继续
❌ 检测到泄漏            → 停止，发我
```

### 提前预期：dense 大概率会赢

`SnowflakeB.forward` **没有跨 token 注意力**：

```python
xsum = xc.mean(1)      # chunk(32字符) 内平均 ← 唯一上下文
hh = matmul(xc, W1_base) + bmm(xA, W1_V)   # 每 token 独立变换
```

dense 有**完整因果注意力（前 127 token）**。所以 README 那句
`a fraction of the parameters of a dense baseline` **几乎确定不成立**。

### 但这不白做 —— 叙事换牌

| ❌ 别这么写 | ✅ 这么写 |
|---|---|
| 参数更少还更好 | Snowflake 用 **0.664x FLOPs** 达到 loss X；dense 需 Y 倍 FLOPs 才达到同等 loss |

而「上下文机制不对等是主因」本身是**有价值的发现**，写进 limitations。

`compare_dense.py` 会自动打印建议的替代表述。

---

## 第 8 步：结果回传（别等最后）

```bash
cd /mnt/workspace/snowflake-moe
tar -czf results_$(date +%m%d_%H%M).tar.gz v2_compositional/output/
ls -lh results_*.tar.gz
```

本地拉取：
```powershell
scp root@<A10>:/mnt/workspace/snowflake-moe/results_*.tar.gz C:\Users\irinna\Desktop\
```

**每跑完一批就传一次** —— 上次服务器释放的教训。

---

## 全流程时间

| 步骤 | 耗时 |
|---|---|
| 0~3 拉取检查 | 2 分钟 |
| 4 CI 审计 | **10 秒** |
| 5 打补丁 | 1 分钟 |
| 6 重跑 10 seeds | **75 分钟** |
| 7 dense 对比 | **50 分钟** |
| **合计** | **约 2.2 小时** |

---

## 常见报错

| 报错 | 原因 | 解法 |
|---|---|---|
| `random_ received from=0 >= to=-129` | 数据为空 | `python3 -c "import os;print(os.path.getsize('data_25mb.txt'))"` 若为 0 → 重建 |
| 参数量 `5,138,432` | vocab=0 的空数据指纹 | 同上 |
| `UnicodeEncodeError: latin-1` | 干净 UTF-8 数据 + `--fix-mojibake` | **去掉 `--fix-mojibake`** |
| `NameError: UnicodeEncodeDecodeError` | 异常名不存在 | `patch_all.py` 已修 |
| `ModuleNotFoundError: modelscope` | 未装 | `pip install modelscope` |
| dense loss ≈ 0.019 | causal mask 泄漏 | 跑 `--leak-check-only` |
| `Permission denied` | 脚本无执行权限 | `chmod +x *.sh` |

---

## 记住两个"指纹"

```
参数量 5,138,432  → vocab=0，数据是空的 ❌
参数量 5,164,132  → vocab=100，数据正常 ✅

dense loss ≈ 0.019 → causal mask 泄漏 ❌
dense loss 1.4~1.8 → 正常 ✅
```

看到异常值直接查对应项，不用读 traceback。
