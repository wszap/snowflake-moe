# Windows 快速上手（PowerShell）

> 你之前用的是 bash 命令，Windows 上跑不了。本文全部换成 PowerShell 命令。

---

## 第 0 步：设置编码（防 GBK 报错）

每次开新 PowerShell 窗口先跑这两行：

```powershell
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## 第 1 步：装 Git

**方式 A（推荐）**
```powershell
winget install --id Git.Git -e --source winget
```
**装完必须重开 PowerShell 窗口**（PATH 才生效）。

**方式 B：winget 不可用**
```powershell
Start-Process "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe"
```
下载完双击安装，一路 Next（默认选项即可）。

**验证**
```powershell
git --version
```
看到 `git version 2.x.x` 就成了。

---

## 第 2 步：一键检查环境 + 修 README

把归档包解压到任意目录（比如 `C:\Users\irinna\snowflake_archive\`），然后：

```powershell
cd C:\Users\irinna\snowflake_archive\scripts
powershell -ExecutionPolicy Bypass -File .\setup_and_restructure.ps1
```

会依次检查：git / python / torch / CUDA / README 五处硬伤。

**若 git 未装**，脚本会提示并问你要不要用 winget 装（`y` 回车即可）。

**检查出硬伤后**，执行 Phase 1 修复：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_and_restructure.ps1 -Phase 1
```

自动改 5 处，改前会备份 README.md。

---

## 第 3 步：推送到 GitHub

```powershell
powershell -ExecutionPolicy Bypass -File .\push_all_to_github.ps1
```

默认推到 `backup/full-archive-MMDD` 分支，**不动 main**。
仓库不存在会自动克隆到归档包同级的 `snowflake-moe\`。

想指定位置：
```powershell
powershell -ExecutionPolicy Bypass -File .\push_all_to_github.ps1 -RepoPath C:\Users\irinna\snowflake-moe
```

---

## 第 4 步：跑 dense baseline（待补的 P0 证据）

```powershell
cd C:\Users\irinna\snowflake-moe\v2_compositional

# ① 先做泄漏自检（几秒）
python dense_baseline.py --leak-check-only

# ② 规模扫描
python dense_baseline.py --data <TinyStories路径> --scale-sweep --max-steps 3000

# ③ 10 seeds
python dense_baseline.py --data <TinyStories路径> --L 5 --d 288 --sweep

# ④ 配对检验
python compare_dense.py --dense output/dense
```

**① 必须先看**：应打印 `✅ 无泄漏（正确 causal）`。
若打印 `❌ 泄漏`，说明 causal mask 还是反的，别往下跑。

---

## 常见报错对照

| 报错 | 原因 | 解决 |
|---|---|---|
| `'git' is not recognized` | git 未装 或 PATH 未生效 | 装完**重开** PowerShell |
| `'export' is not recognized` | 用了 bash 命令 | 改 PowerShell 语法 |
| `The system cannot find the file specified` | 路径含中文/空格 或 文件不存在 | 用引号包裹路径 |
| `UnicodeEncodeError: 'gbk'` | Windows 默认 GBK | 跑第 0 步那两行 |
| `running scripts is disabled` | 执行策略限制 | 加 `-ExecutionPolicy Bypass` |
| `无法加载文件 ... 未数字签名` | 同上 | 同上 |

---

## 路径含空格怎么办

PowerShell 里路径要加引号：

```powershell
cd "C:\Users\irinna\My Documents\snowflake_archive\scripts"
powershell -ExecutionPolicy Bypass -File ".\push_all_to_github.ps1" -RepoPath "C:\My Repos\snowflake-moe"
```

---

## 归档包结构

```
SNOWFLAKE_ARCHIVE\
├── MIGRATION_GUIDE.md      迁移手册
├── WINDOWS_QUICKSTART.md   本文
├── v2_compositional\       21 个文件（代码 + RESULTS.md）
├── docs\                   6 个文件（博客/负结果/实验日志）
├── scripts\
│   ├── push_all_to_github.ps1      ★ Windows 推送
│   ├── setup_and_restructure.ps1   ★ Windows 环境+整改
│   ├── push_all_to_github.sh       Linux/Mac 推送
│   ├── restructure_repo.sh         Linux/Mac 整改
│   └── collect_remote_evidence.sh  服务器取证
└── _remote_evidence\       远端证据占位 + 10 seeds 数据表
```

---

## 记得做：远端服务器取证

**10 seeds 的 `record.json` + `steps.csv` 只在服务器上，本地没有。**

在【服务器】上跑（Linux）：
```bash
bash collect_remote_evidence.sh
```
然后下载：
```powershell
scp root@<服务器>:~/snowflake_remote_backup_*.tar.gz C:\Users\irinna\
```

（Windows 没装 scp 的话，用 WinSCP 或 FileZilla 图形化下载。）
