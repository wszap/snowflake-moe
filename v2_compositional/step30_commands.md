# V1.005 命令清单（DSW）

## 只传一个文件

把 `step30_install.sh` 上传到 `/mnt/workspace/TEMP-FILE-STATION/`（`.sh` 不会被浏览器加 `.ZIP`）。

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
bash /mnt/workspace/TEMP-FILE-STATION/step30_install.sh 2>&1 | tee step30_install.log
```

必须看到：

```
PATCH_OK  patches=N  new_files=5
FORMULA_SELFTEST=PASS
INSTALL_STEP30_OK
```

锚点不唯一会 `FATAL` 停线并打印实际命中次数，**不覆盖任何文件**；备份 `*.bak_step30`。
公式自检会打印「16 主干组 + 16 gate 组 / step0 k 全 1 / 主干 lr 真分化」。

## 三关（别一次粘完）

### ① 第一关 SMOKE 20 步（约 2 分钟，6 档）

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
bash step30_ab.sh 2>&1 | tee step30_smoke_run.log
grep -E '##########|exit=|json=|SMOKE_|STEP30_DONE' step30_smoke_run.log
```

必须 `STEP30_DONE_OK` **且** `SMOKE_OK=1`：
- **G1** off 档零 trace；同配置 `st30_ref_s20_nopatch` 逐位全 0（配置不同只 SKIP）
- **G2 ★ 承重**：`bb_*` 每步 `n_bb_groups=16` 且 `bb_lr` 真分化；`gate_cat/lrx/A0` 为 0 组
- **T1** `f_std>0.05`、**T2** `eΔ>0.10`、**T3** 活跃窗 `corr>0.4`

`SMOKE_OK=0` ⇒ 停线，贴日志；回滚：

```bash
cp train_bio.py.bak_step30 train_bio.py
cp paperplane_moe/enzyme_rate.py.bak_step30 paperplane_moe/enzyme_rate.py
```

### ② 第二关 SHORT 300 步（只有①过了才跑）

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
nohup env SHORT=1 bash step30_ab.sh > step30_short.log 2>&1 & echo "PID=$!"
grep -E '##########|exit=|json=|STEP30_DONE' step30_short.log
python3 step30_reach_summary.py 2>&1 | tee step30_short_summary.log
```

**注意**：`THR=1.30` 是给 FULL/3000 步校准的，300 步 `floor≈2.20` ⇒ `T_ε` 全 None 是**量程**不是负结果；SHORT 只看 `[1] G2` 与 `[4] κ 代理` 的方向。等 `STEP30_DONE_OK` 且 6 个 `_s300` json 齐再跑判据。

### ③ 第三关 FULL 3000 步（只有②有可读方向才跑）

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
nohup env FULL=1 bash step30_ab.sh > step30_full.log 2>&1 & echo "PID=$!"
```

阻塞等待（含「假完成」检测；`step30_ab.sh` 无 `set -e`，DONE 但不全齐要停）：

```bash
cd /mnt/workspace/snowflake-moe/v2_compositional
TAGS="st30_ref_nopatch st30_A0_B st30_gate_cat st30_bb_cat st30_bb_const st30_lrx st30_bb_freeze"
for i in $(seq 1 400); do
  H=$(awk '/##########/{n++} END{print n+0}' step30_full.log 2>/dev/null)
  E=$(awk '/exit=/{n++} END{print n+0}' step30_full.log 2>/dev/null)
  BAD=$(awk '/exit=/ && !/exit=0/{n++} END{print n+0}' step30_full.log 2>/dev/null)
  D=$(awk '/STEP30_DONE/{n++} END{print n+0}' step30_full.log 2>/dev/null)
  M=0; for t in $TAGS; do [ -s "output/v4/$t.json" ] && M=$((M+1)); done
  PROC=$(ps -eo comm=,args= | awk '$1 ~ /^python/ && /train_bio\.py/{n++} END{print n+0}')
  BASH=$(ps -eo comm=,args= | awk '$1 ~ /^bash/ && /step30_ab\.sh/{n++} END{print n+0}')
  if [ "$D" -ge 1 ] && [ "$H" -gt 0 ] && [ "$H" -eq "$E" ] && [ "$BAD" -eq 0 ] && [ "$M" -ge "$H" ]; then
    echo "【完成且完整】H=$H E=$E BAD=$BAD json=$M"; break
  fi
  if [ "$D" -ge 1 ]; then echo "【假完成】DONE 但 H=$H E=$E BAD=$BAD json=$M → 停"; break; fi
  if [ "$PROC" -eq 0 ] && [ "$BASH" -eq 0 ]; then echo "【异常】进程没了却没 DONE → 停"; break; fi
  [ $((i % 8)) -eq 0 ] && echo "  ...H=$H E=$E json=$M ($((i*15))s)"
  sleep 15
done
grep -E '##########|exit=|STEP30_DONE' step30_full.log
python3 step30_reach_summary.py 2>&1 | tee step30_full_summary.log
```

7 档 × 3000 步；v10 单档实测约 32 分钟，留足 100 分钟。**若 pod 再换**，重跑同一条 `step30_ab.sh` 会覆盖同名 tag —— 不建议；改用 `SEED=1` 或把已落盘 json 改名留档后单独补缺档。

## 判读（跑前写死）

| 结果 | 写法 |
|---|---|
| `[0] FAIL` | 回滚 `*.bak_step30`，停线 |
| `[1] G2 FAIL` | 主干没吃到 k_i，实验等于没做，贴日志 |
| `[3] G3 PASS` 且 `T_ε(bb_cat)` 明显更小 | **速率常数型酶在拼专家里第一次成立**：同时赢过等效 lr 两个对照 |
| `[3] G3 FAIL` / 打平 | 与等效 lr 不可分，如实收口，**不调 β/Km 硬凑** |
| `gate_cat−A0` 打平 | 复现 step29e 的作用面因果（阴性对照成立） |

单 seed 仅方向、不叫显著；终态 val 只是门禁；**永远不写「加速」**（wall-clock / avg_step 三档对比不在判据内）。

## 边界

- 沙盒无 torch：安装器里的 `FORMULA_SELFTEST` 是纯标准库 fake 对象，只验「分组数 / k0≡1 / lr 真分化」，**真 301 点零污染与真收益只能在 DSW 出**。
- 这轮不改前向、不改参数量、不碰 RNG；`A0_B off` 档与未补丁版逐位一致由 U3 实证。
- 已归档结论一字不翻案：`V10−B=+0.1099`、`B−A=+0.0002`、`C−B≈−0.0003`、`grouped−global=+0.0059`、`warm−zero=−0.0003`、B5 `p≈0.09` / 翻盖 `p≈0.083`；**显著成效未达成**。
- 推 GitHub 由你在服务器 `git add && commit && push`；我无写凭据。
