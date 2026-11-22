# step29 / V1.004 · DSW 侧命令清单

> 全部命令都在 `/mnt/workspace/snowflake-moe/v2_compositional` 下执行。
> 只复制**代码块**，不要把中文正文一起粘进终端（前面几轮的 `-bash: 缺: 未找到命令` 就是这么来的）。

---

## 0 · 上传 + 安装（只做一次）

把 `step29_install.sh`（单个 `.sh`，不会被浏览器加 `.ZIP` 后缀）上传到
`/mnt/workspace/TEMP-FILE-STATION/`，然后：

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
bash /mnt/workspace/TEMP-FILE-STATION/step29_install.sh 2>&1 | tee step29_install.log
```

**装好标志**（缺一不可）：

```
INSTALL_STEP29_OK
ast.parse 5/5 OK
bash -n OK
FORMULA_SELFTEST=PASS
锚点命中: compose=2  train_bio=9
```

脚本做五件事：备份 `*.bak_step29` → 落盘 5 个新文件 → 对 `compose.py` /
`train_bio.py` 打 **11 处精确锚点补丁**（命中次数必须各为 1，否则停，不强覆盖）
→ 静态 + 公式自检 → 打印下一步。**不跑任何训练、不 push GitHub。**

可选复核（补丁只给 `compose.py` 加了两行 detach 统计，单测应原样通过）：

```bash
python3 verify_bio.py 2>&1 | tee step29_verify_bio.log
grep -E 'PASS|FAIL 清单' step29_verify_bio.log | tail -12
```

若报「锚点命中 N 次（应为 1）」，把 `step29_install.log` 全文贴回来；
**不要手工改文件**，我按真实内容出第二版补丁。

---

## 1 · 第一关：20 步 smoke（约 2 分钟，4 档）

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
bash step29_ab.sh 2>&1 | tee step29_smoke_run.log
grep -E '##########|exit=|json=|SMOKE_|STEP29_DONE' step29_smoke_run.log
```

必须同时看到 `STEP29_DONE_OK` 和 `SMOKE_OK=1`。判据是预先写死的三条单元测试：

| 项 | 要求 | 对应文献 |
|---|---|---|
| T1 | 活跃窗口 `std(f)>0.05` 且 `0.05<median(f)<0.95` | MM 一级/零级区允许 f 触及 0/1；只验 Km 没让 f 退化成常数 |
| T2 | `mean\|e_end−e_start\| > 0.10` | Jacob–Monod 诱导/降解慢变量真的在动 |
| T3 | `corr(k, 1/H) > 0.5` | MCA：酶压在当下最慢的步骤上（只要求 cat / cat_het，const/lrx 的 k 恒定是设计使然） |
| U2/U3 | off 档 trace 为空、k 全 1、与 `st25_s0_B` 共同点逐位相等 | 零污染硬前置的快速哨兵 |

**`SMOKE_OK=0` ⇒ 停线**，别起第二关。回滚：

```bash
cp train_bio.py.bak_step29 train_bio.py
cp paperplane_moe/compose.py.bak_step29 paperplane_moe/compose.py
```

---

## 2 · 第二关：300 步短档（约 25 分钟，7 档）

只有第一关过了才跑：

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
nohup env SHORT=1 bash step29_ab.sh > step29_short.log 2>&1 & echo "PID=$!"
```

单独轮询（**不要一次粘到底**）：

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
grep -E '##########|exit=|json=|STEP29_DONE' step29_short.log
python3 step29_reach_summary.py 2>&1 | tee step29_short_summary.log
```

第二关的 `T_ε` 阈值默认 1.30；300 步可能还没到，判据会如实打印 `None`，
**不要为了凑数改 `STEP29_THR`**。这一关看的是：κ(H) vs κ(k·H)、cat 与
`const`/`lrx` 是否开始分开、以及分裂（step 150）后 `rebind_groups` 有没有生效
（看 `k_std` 在 150 之后是否还保持非零）。

---

## 3 · 第三关：3000 步正式八档（约 2.5 小时）

只有第二关的 `cat vs const/lrx` 已有可读方向才跑：

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
nohup env FULL=1 bash step29_ab.sh > step29_full.log 2>&1 & echo "PID=$!"
```

**阻塞等待（自己等到齐才出判据，别手动轮询；已含「假完成」检测）**：

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
for i in $(seq 1 600); do
  H=$(awk '/##########/{n++} END{print n+0}' step29_full.log 2>/dev/null)
  E=$(awk '/exit=/{n++} END{print n+0}' step29_full.log 2>/dev/null)
  BAD=$(awk '/exit=/ && !/exit=0/{n++} END{print n+0}' step29_full.log 2>/dev/null)
  D=$(awk '/STEP29_DONE/{n++} END{print n+0}' step29_full.log 2>/dev/null)
  M=$(awk '/json=OK/{n++} END{print n+0}' step29_full.log 2>/dev/null)
  PROC=$(ps -eo comm=,args= | awk '$1 ~ /^python/ && /train_bio\.py/{n++} END{print n+0}')
  if [ "$D" -ge 1 ] && [ "$H" -gt 0 ] && [ "$H" -eq "$E" ] && [ "$BAD" -eq 0 ] && [ "$M" -ge "$H" ]; then
    echo "【完成且完整】H=$H E=$E BAD=$BAD json=$M"; break
  fi
  if [ "$D" -ge 1 ]; then echo "【假完成】DONE 但 H=$H E=$E BAD=$BAD json=$M → 停，贴出来"; break; fi
  if [ "$PROC" -eq 0 ] && [ "$H" -gt 0 ]; then echo "【异常】进程没了却没 DONE → 停"; break; fi
  [ $((i % 8)) -eq 0 ] && echo "  ...H=$H E=$E json=$M ($((i*15))s)"
  sleep 15
done
python3 step29_reach_summary.py 2>&1 | tee step29_full_summary.log
```

---

## 4 · 档位对照表

| tag | `--enz-rate-mode` | 角色 |
|---|---|---|
| `st29_A0_B` | off + `--cell-type B --zero-check` | 301 点零污染硬前置 |
| `st29_A1_res` | off + residual | 拼专家基线（v9 的 A0） |
| `st29_amp_old` | amp + `--bio-on` | V1.003 旧实现（`ci·term` 乘前向）对照 |
| `st29_cat` | cat | ★ 主实验：逐 cell 底物诱导的速率常数 |
| `st29_cat_het` | cat_het | 每 cell Km 自主漂变（「各种酶」） |
| `st29_const` | const | 消融：k 取全局平均 |
| `st29_lrx` | lrx | 二次消融：不分组，lr×k̄ |
| `st29_cat_freeze` | cat + `--enz-freeze-cell 0` | 速率 rescue |

---

## 5 · 判读口径（跑之前写死）

| 结果 | 写法 |
|---|---|
| `[0]` 硬前置 FAIL | 回滚 `*.bak_step29`，停线 |
| `[1]` 任一 cat 档不过 | 「酶还没上场，实验未测到」，不是「酶没用」 |
| `[3]` cat 同时显著早于 const / lrx | 速率常数型酶在拼专家里成立，**单 seed 仅方向，不叫显著** |
| `[3]` FAIL 或 T_ε 打平 | 与等效 lr 不可分，如实收口，**不调 β/Km 硬凑** |
| 任一档终态 val 更差 | 如实收口，终态 val 只是「不劣于」门禁 |

**永远不写「加速」**：wall-clock / `avg_step` 三档对比不在判据内，终点等价不等于提速。

**体积提醒**：`enz_rate_trace` 是每步全字段（8 个 cell × 7 组数值），
`indent=2` 下 3000 步约 4~6 MB/档；FULL 关 6 个 cat 系档合计约 30 MB，
写盘可接受，但别再额外复制 tag。

---

## 6 · 归档（由你执行，我无写凭据）

```bash
cd /mnt/workspace/snowflake-moe
git add -A && git add -f '*.log'
git commit -m "step29 V1.004: enzyme as per-cell parameter-update rate constant (MM substrate = g/H)"
git push origin HEAD:main && git push origin HEAD:backup/full-archive-1017
git rev-parse --short HEAD
```

把新短 SHA 发我，我重打一份与线上完全一致的快照。

---

## 7 · 不翻案清单

`V10−B=+0.1099`、`grouped−global=+0.0059`、`warm−zero=−0.0003`、
`B−A=+0.0002`、`C−B≈−0.0003`、B5 `p≈0.09` / 翻盖 `p≈0.083` 一字不翻案。
step28 的零收益现在有了统一解释：**酶乘在产物上 + 卷子只考终态**，不是酶原语无效。
v9/v9.1/v9.2 的 toy 结论不外推到真模型；显著成效未达成。
