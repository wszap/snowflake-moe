# ════════════════════════════════════════════════════════════════
#  Windows 环境一键配置 + 仓库整改 (PowerShell 版)
#
#  用法:
#     powershell -ExecutionPolicy Bypass -File .\setup_and_restructure.ps1
#     powershell -ExecutionPolicy Bypass -File .\setup_and_restructure.ps1 -Phase 1
#
#  -Phase 0  只做环境检查（默认）
#  -Phase 1  修 v1.0 README 五处硬伤
# ════════════════════════════════════════════════════════════════
param(
    [int]$Phase = 0,
    [string]$RepoPath = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-OK   { param($m) Write-Host "[OK]   $m" -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Write-Err  { param($m) Write-Host "[ERR]  $m" -ForegroundColor Red }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ArchiveDir = Split-Path -Parent $ScriptDir

Write-Host "═══════════════════════════════════════════"
Write-Host "  Snowflake MoE — Windows 环境配置与整改"
Write-Host "═══════════════════════════════════════════"
Write-Host ""

# ── 检查 git ───────────────────────────────────────────────────
Write-Host "[1/4] 检查 git"
$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
    Write-OK "git 已安装: $(git --version)"
} else {
    Write-Warn "git 未安装"
    Write-Host ""
    Write-Host "  请任选一种安装："
    Write-Host ""
    Write-Host "  A) winget (推荐):"
    Write-Host "     winget install --id Git.Git -e --source winget"
    Write-Host ""
    Write-Host "  B) 下载安装包:"
    Write-Host "     Start-Process 'https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe'"
    Write-Host ""
    Write-Host "  ⚠ 装完必须【重开 PowerShell】再运行本脚本"
    Write-Host ""
    $ans = Read-Host "  现在尝试用 winget 安装? (y/n)"
    if ($ans -eq "y") {
        winget install --id Git.Git -e --source winget
        Write-Host ""
        Write-Host "  ⚠ 请重开 PowerShell 后重新运行本脚本"
        exit 0
    } else {
        exit 1
    }
}

# ── 检查 python ────────────────────────────────────────────────
Write-Host ""
Write-Host "[2/4] 检查 python"
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python3 -ErrorAction SilentlyContinue }
if ($py) {
    Write-OK "python: $(& $py.Source --version 2>&1)"
    $torch = & $py.Source -c "import torch; print(torch.__version__)" 2>$null
    if ($torch) {
        Write-OK "torch: $torch"
        $cuda = & $py.Source -c "import torch; print(torch.cuda.is_available())" 2>$null
        Write-OK "CUDA available: $cuda"
    } else {
        Write-Warn "torch 未安装，跑训练前需: pip install torch"
    }
} else {
    Write-Warn "python 未找到"
}

# ── 定位仓库 ───────────────────────────────────────────────────
Write-Host ""
Write-Host "[3/4] 定位仓库"
if ([string]::IsNullOrEmpty($RepoPath)) {
    $RepoPath = Join-Path (Split-Path -Parent $ArchiveDir) "snowflake-moe"
}
if (-not (Test-Path (Join-Path $RepoPath ".git"))) {
    Write-Warn "仓库不存在: $RepoPath"
    Write-Host "  先运行 push_all_to_github.ps1（会自动克隆）"
} else {
    Write-OK "仓库: $RepoPath"
}

# ── Phase 0: 检查 README 硬伤 ──────────────────────────────────
Write-Host ""
Write-Host "[4/4] README 硬伤检查"
$readme = Join-Path $RepoPath "README.md"
if (-not (Test-Path $readme)) {
    Write-Warn "README.md 不存在"
} else {
    $content = Get-Content $readme -Raw -Encoding UTF8
    $issues = @()

    if ($content -match "python run_lm\.py") {
        Write-Warn "① 坏命令: 'python run_lm.py'（文件不存在）"
        $issues += "run_lm"
    } else { Write-OK "① 坏命令已修" }

    if ($content -match "OOD Generalization") {
        Write-Warn "② OOD 标题需改（ratio<1，OOD 集更简单）"
        $issues += "ood"
    } else { Write-OK "② OOD 标题已改" }

    if ($content -match "Parameter Efficiency") {
        Write-Warn "③ Parameter Efficiency 需改标题（是 active/fixed，非 vs dense）"
        $issues += "pe"
    } else { Write-OK "③ 标题已改" }

    if ($content -match "fraction of the parameters of a dense baseline") {
        Write-Warn "④ 'a fraction of dense baseline' 无数据支撑"
        $issues += "dense"
    } else { Write-OK "④ 已处理" }

    if ($content -match "memory-value slot") {
        Write-Warn "⑤ 忆点定义需加版本标注"
        $issues += "mp"
    } else { Write-OK "⑤ 已标注" }

    Write-Host ""
    if ($issues.Count -gt 0) {
        Write-Host "  发现 $($issues.Count) 处待修。运行："
        Write-Host "     powershell -ExecutionPolicy Bypass -File `"`"$ScriptDir\setup_and_restructure.ps1`"`" -Phase 1 -RepoPath `"$RepoPath`""
    } else {
        Write-OK "无待修项"
    }

    # ── Phase 1: 修复 ──────────────────────────────────────────
    if ($Phase -eq 1 -and $issues.Count -gt 0) {
        Write-Host ""
        Write-Host "═══ 执行 Phase 1 修复 ═══"
        $bak = $readme + ".bak." + (Get-Date -Format "yyyyMMddHHmmss")
        Copy-Item $readme $bak
        Write-OK "已备份: $bak"

        $c = $content

        if ($issues -contains "run_lm") {
            $c = $c -replace "python run_lm\.py # via run_experiments\.py",
                             "python run_experiments.py   # LM scaling & ablation sweep`npython train_lm.py           # 基础 LM 训练"
            $c = $c -replace "python run_lm\.py", "python run_experiments.py"
            Write-OK "① 坏命令已修"
        }

        if ($issues -contains "ood") {
            $c = $c -replace "### OOD Generalization \(Shakespeare → OOD\)",
                "### OOD Robustness — Relative Degradation (Shakespeare → OOD)`n`n> **注意**：两个模型的 ood/id ratio 均 < 1，说明 OOD 评测集本身比 ID 集更简单。`n> 因此这里衡量的不是「OOD 泛化有多好」，而是**同等分布偏移下的相对退化程度**。"
            Write-OK "② OOD 标题已改"
        }

        if ($issues -contains "pe") {
            $c = $c -replace "### Parameter Efficiency \(Snowflake v3, Shakespeare\)",
                "### Sparse Activation Ratio (Snowflake v3, Shakespeare)`n`n> **注意**：Param Ratio = Active / Fixed（稀疏激活比），**不是**与 dense baseline 的对比。`n> 本仓库目前**尚未包含 dense baseline 对照**（v2.0 正在补充，见 `v2_compositional/`）。"
            Write-OK "③ 标题已改"
        }

        if ($issues -contains "dense") {
            $c = $c -replace "achieving strong generalization under a fraction of the parameters of a dense baseline\.",
                "achieving strong generalization under a fraction of the parameters of a dense baseline.`n  <sub>⚠️ 该 claim 目前**尚无 dense baseline 对照数据**。v2.0 已实现 dense 对照脚本（`v2_compositional/dense_baseline.py` + `compare_dense.py`），结果待补。</sub>"
            Write-OK "④ 已加标注"
        }

        if ($issues -contains "mp") {
            $c = $c -replace "\*\*Memory Points \(忆点\)\*\*: A dedicated memory-value slot mechanism that stores decoupled factual knowledge, separate from routing weights\.",
                "**Memory Points (忆点)**: A dedicated memory-value slot mechanism that stores decoupled factual knowledge, separate from routing weights.`n  <sub>**v1.0 定义**：忆点 = 独立值槽，与路由分离，可单独擦除。<br>`n  **v2.0 定义**（见 `v2_compositional/`）：忆点 = 连接本体，与路由耦合。<br>`n  两者是**不同架构**下的不同定义，请勿混用。</sub>"
            Write-OK "⑤ 已加版本标注"
        }

        # 用 UTF8 无 BOM 写回
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($readme, $c, $utf8NoBom)
        Write-OK "README.md 已更新（UTF-8 无 BOM）"
    }
}

Write-Host ""
Write-Host "═══ 完成 ═══"
