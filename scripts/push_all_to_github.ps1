# ════════════════════════════════════════════════════════════════
#  一键推送到 GitHub (Windows PowerShell 版)
#
#  用法:
#     powershell -ExecutionPolicy Bypass -File .\push_all_to_github.ps1
#     powershell -ExecutionPolicy Bypass -File .\push_all_to_github.ps1 -RepoPath C:\path\to\snowflake-moe
#
#  默认推到 backup/full-archive-MMDD 分支，【不动 main】
# ════════════════════════════════════════════════════════════════
param(
    [string]$RepoPath = "",
    [switch]$ToMain
)

$ErrorActionPreference = "Stop"

# UTF-8 输出，避免 GBK 报错
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-OK   { param($m) Write-Host "[OK]   $m" -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "[WARN] $m" -ForegroundColor Yellow }
function Write-Err  { param($m) Write-Host "[ERR]  $m" -ForegroundColor Red; exit 1 }

# ── 0. 定位归档目录 ────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ArchiveDir = Split-Path -Parent $ScriptDir

Write-Host "═══ 推送配置 ═══"
Write-Host "  归档源: $ArchiveDir"

# ── 1. 检查 git ────────────────────────────────────────────────
$git = Get-Command git -ErrorAction SilentlyContinue
if (-not $git) {
    Write-Err "未找到 git。请先安装：winget install --id Git.Git -e --source winget`n装完重开 PowerShell。"
}
Write-OK "git: $($git.Source)"

# ── 2. 定位/克隆仓库 ───────────────────────────────────────────
if ([string]::IsNullOrEmpty($RepoPath)) {
    $RepoPath = Join-Path (Split-Path -Parent $ArchiveDir) "snowflake-moe"
    Write-Warn "未指定 RepoPath，使用默认: $RepoPath"
}

if (-not (Test-Path (Join-Path $RepoPath ".git"))) {
    Write-Host "  仓库不存在，正在克隆..."
    $parent = Split-Path -Parent $RepoPath
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    Push-Location $parent
    git clone https://github.com/wszap/snowflake-moe.git
    Pop-Location
    if (-not (Test-Path (Join-Path $RepoPath ".git"))) {
        Write-Err "克隆失败，请检查网络或手动克隆到 $RepoPath"
    }
}
Write-OK "仓库: $RepoPath"

Push-Location $RepoPath

# ── 3. 分支 ────────────────────────────────────────────────────
if ($ToMain) {
    git checkout main 2>$null
    if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Err "无法切到 main" }
    git pull --ff-only 2>$null
    $Branch = "main"
} else {
    $Branch = "backup/full-archive-" + (Get-Date -Format "MMdd")
    git checkout -B $Branch
    if ($LASTEXITCODE -ne 0) { Pop-Location; Write-Err "无法建分支 $Branch" }
    Write-OK "分支就绪: $Branch"
}

# ── 4. 同步文件 ────────────────────────────────────────────────
Write-Host ""
Write-Host "═══ 同步文件 ═══"

# v2_compositional
$dst = Join-Path $RepoPath "v2_compositional"
New-Item -ItemType Directory -Path $dst -Force | Out-Null
Copy-Item "$ArchiveDir\v2_compositional\*" -Destination $dst -Recurse -Force
Write-OK "v2_compositional/ ← $((Get-ChildItem "$ArchiveDir\v2_compositional" -File).Count) 个文件"

# docs
$dst = Join-Path $RepoPath "docs"
New-Item -ItemType Directory -Path $dst -Force | Out-Null
Copy-Item "$ArchiveDir\docs\*" -Destination $dst -Recurse -Force
Write-OK "docs/ ← $((Get-ChildItem "$ArchiveDir\docs" -File).Count) 个文件"

# scripts（排除自身和 .sh）
$dst = Join-Path $RepoPath "scripts"
New-Item -ItemType Directory -Path $dst -Force | Out-Null
Get-ChildItem "$ArchiveDir\scripts" -File | Where-Object {
    $_.Extension -ne ".sh" -and $_.Name -ne "push_all_to_github.ps1"
} | ForEach-Object { Copy-Item $_.FullName -Destination $dst -Force }
# 但保留 .sh 供 Linux/Mac 用
Get-ChildItem "$ArchiveDir\scripts" -Filter "*.sh" | ForEach-Object {
    Copy-Item $_.FullName -Destination $dst -Force
}
Write-OK "scripts/ 已同步"

# 根级文档
Copy-Item "$ArchiveDir\scripts\README_main.md" -Destination (Join-Path $RepoPath "README_main.md") -Force
Copy-Item "$ArchiveDir\scripts\README_v1.md"   -Destination (Join-Path $RepoPath "README_v1.md") -Force
Copy-Item "$ArchiveDir\scripts\MARVIS_TASK.md" -Destination (Join-Path $RepoPath "MARVIS_TASK.md") -Force
Copy-Item "$ArchiveDir\scripts\INTEGRATION_PLAN.md" -Destination (Join-Path $RepoPath "INTEGRATION_PLAN.md") -Force
Write-OK "根级文档已同步"

# ── 5. 体积检查 ────────────────────────────────────────────────
Write-Host ""
Write-Host "═══ 体积检查 ═══"
$big = Get-ChildItem -Path $RepoPath -Recurse -File -ErrorAction SilentlyContinue |
       Where-Object { $_.FullName -notmatch "\\\.git\\" -and $_.Length -gt 10MB }
if ($big) {
    Write-Warn "以下文件 >10MB，GitHub 可能拒收："
    $big | ForEach-Object { Write-Host "    $($_.FullName)" }
} else {
    Write-OK "无超大文件"
}

# ── 6. 提交 ────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══ 提交 ═══"
git add -A
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Warn "无改动可提交"
} else {
    git commit -m "archive: full v2.0 integration package (pre-migration backup)

- v2_compositional/: SnowflakeB + train_lock60 + run_metrics + dense baseline
  + compare_dense + verify_six_traps + RESULTS.md (10 seeds, 5 hypotheses PASS)
- docs/: blog (six MoE traps), NEGATIVE_RESULTS, EXPERIMENT_LOG (37 decisions)
- scripts/: restructure_repo.sh, collect_remote_evidence.sh, MARVIS_TASK.md
- README_main.md / README_v1.md: dual-version navigation + known limitations

Purpose: full backup before machine migration."
    Write-OK "已提交"
}

# ── 7. 推送 ────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══ 推送 ═══"
$remote = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0 -and $remote) {
    git push -u origin $Branch
    if ($LASTEXITCODE -eq 0) {
        Write-OK "已推送到 origin/$Branch"
    } else {
        Write-Warn "推送失败。可能需要登录，或手动执行：git push -u origin $Branch"
    }
} else {
    Write-Warn "无 origin remote，未推送。手动添加："
    Write-Host "     git remote add origin https://github.com/wszap/snowflake-moe.git"
}

Pop-Location

Write-Host ""
Write-Host "═══ 完成 ═══"
Write-Host "  分支: $Branch"
Write-Host ""
Write-Host "下一步（换电脑后）:"
Write-Host "  1. git clone + git checkout $Branch"
Write-Host "  2. 按 MARVIS_TASK.md 执行仓库整改"
Write-Host "  3. 跑 dense baseline 补 P0 证据"
