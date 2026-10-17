# 本地 Windows → GitHub → A10 完整链路

> 安全检查已完成：**归档包可以全部公开推送**
> 无需走 ssh 的敏感内容（无密钥/无 IP/无数据集/无 ckpt）

---

## 安全扫描结论

| 检查项 | 结果 |
|---|---|
| 密钥 / Token / 密码 / API Key | ✅ 无 |
| IP 地址 | ✅ 无 |
| ssh 私钥 / .env / .pem | ✅ 无 |
| 数据集（100MB） | ✅ 不在包内 |
| ckpt | ✅ 不在包内 |
| 个人信息 | ⚠️ 仅 `C:\Users\irinna` 示例路径 6 处（可选替换） |

**→ 整个包可公开推送。ssh 只用于后续传数据集 / 拉结果。**

---

## 第 1 步：本地装 Git（若已装跳到第 2 步）

```powershell
winget install --id Git.Git -e --source winget
```

**装完必须重开 PowerShell 窗口**（PATH 才生效）。

winget 不可用时，直接下安装包：
```powershell
Start-Process "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe"
```

验证：
```powershell
git --version
```

**git 实在装不上 → 走第 1-B 步的网页上传方案。**

---

## 第 2 步：配置 git 身份（首次必做）

```powershell
git config --global user.email "you@example.com"
git config --global user.name "wszap"
```

---

## 第 3 步：设置编码（防 GBK 报错）

```powershell
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## 第 4 步：推送（选一种）

### 方案 A：脚本自动（推荐）

```powershell
cd <归档包解压目录>\scripts
powershell -ExecutionPolicy Bypass -File .\push_all_to_github.ps1 -RepoPath C:\Users\irinna\snowflake-moe
```

脚本会：克隆仓库 → 建 `backup/full-archive-MMDD` 分支 → 复制文件（**跳过已存在的 LICENSE/requirements 等，不覆盖**）→ 提交 → 推送。

### 方案 B：手动命令（想看清每一步）

```powershell
cd C:\Users\irinna
git clone https://github.com/wszap/snowflake-moe.git
cd snowflake-moe

# 确认原历史完整（应看到你的 18 个 commit）
git log --oneline | head -20

# 建独立分支，main 不动
git checkout -B backup/full-archive-1017

# 只新增，不覆盖（注意目标目录是新建的 v2_compositional）
Copy-Item -Recurse -Force <解压目录>\v2_compositional .\v2_compositional
Copy-Item -Recurse -Force <解压目录>\docs\* .\docs\
New-Item -ItemType Directory -Force .\scripts
Copy-Item -Recurse -Force <解压目录>\scripts\* .\scripts\
Copy-Item <解压目录>\scripts\README_main.md .\README_v2_main.md
Copy-Item <解压目录>\scripts\README_v1.md .\README_v1_memory_points.md
Copy-Item <解压目录>\A10_REBUILD.md .
Copy-Item <解压目录>\DATA_RECOVERY.md .
Copy-Item <解压目录>\MIGRATION_GUIDE.md .
Copy-Item <解压目录>\WINDOWS_QUICKSTART.md .

# ★ 关键：确认没有删除任何原有文件
git status --short
# 应只看到 A（新增）和 M（docs 并入），不应看到 D（删除）

git add -A
git commit -m "archive: add v2.0 compositional results (branch backup, main untouched)"
git push -u origin backup/full-archive-1017
```

### 方案 C：git 装不上 → 网页上传

1. 打开 `https://github.com/wszap/snowflake-moe`
2. 切到 `backup/full-archive-1017` 分支（或新建）
3. `Add file` → `Upload files`
4. 把 `v2_compositional\`、`docs\`、`scripts\` 整个文件夹拖进去
5. 填 commit message → `Commit changes`

**缺点**：文件多时要点多次；且无法自动跳过冲突文件。

---

## 第 5 步：A10 上拉取

```bash
# 克隆并切到备份分支
git clone -b backup/full-archive-1017 https://github.com/wszap/snowflake-moe.git
cd snowflake-moe

# 确认内容到位
ls v2_compositional/    # 应有 23 个文件
ls docs/                # 应有 6+ 个文件
```

---

## 第 6 步：A10 上运行

```bash
cd snowflake-moe/v2_compositional

export DATA=/root/tinystories_100mb.txt

# ① 环境 + 下载数据
bash ../scripts/a10_rebuild.sh --setup

# ② 泄漏自检（必做）
bash ../scripts/a10_rebuild.sh --verify
#   必须看到 ✅ 无泄漏（正确 causal）

# ③ 摸底
bash ../scripts/a10_rebuild.sh --smoke

# ④ ablation
bash ../scripts/a10_rebuild.sh --ablation

# ⑤ 10 seeds
bash ../scripts/a10_rebuild.sh --seeds

# ⑥⑦ dense baseline（新证据）
bash ../scripts/a10_rebuild.sh --dense-sweep
bash ../scripts/a10_rebuild.sh --dense-seeds
```

**⚠ A10 上不要加 `--no-bf16`**（DCU 才需要）。脚本里已经处理好了。

---

## ⚠ 注意事项

**① `git status --short` 别跳过**
只应有 `A`（新增）和 `M`（docs 并入）。
出现 `D`（删除）说明覆盖出问题，立即停。

**② main 分支始终不动**
所有内容都在 `backup/full-archive-*` 分支。
确认无误后再决定要不要合并到 main。

**③ GitHub 推送认证**
现在 GitHub 不再接受密码，需要 Personal Access Token：
- GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
- 勾选 `repo` 权限 → Generate token
- 推送时密码栏粘贴这个 token

**④ 数据集走 ssh/scp（公开源下载慢时）**
```powershell
# 本地下载后传到 A10
scp C:\Users\irinna\Downloads\tinystories_100mb.txt root@<A10>:/root/
```

**⑤ 结果回传走 scp**
```powershell
scp -r root@<A10>:~/snowflake-moe/v2_compositional/output/ C:\Users\irinna\results\
```

---

## 检查清单

- [ ] `git --version` 成功
- [ ] git 身份已配置
- [ ] `git log --oneline` 看到原有 18 个 commit
- [ ] `git status --short` 无 `D`（删除）
- [ ] 已推送到 `backup/full-archive-1017`
- [ ] A10 上 `git clone -b backup/full-archive-1017` 成功
- [ ] A10 上 `ls v2_compositional/` 有 23 个文件
- [ ] `--verify` 显示 ✅ 无泄漏
