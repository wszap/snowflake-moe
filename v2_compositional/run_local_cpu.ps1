param(
    [string]$Python = "python",
    [string]$Data = "data_25mb.txt",
    [int]$Steps = 200,
    [int]$Threads = 2,
    [int]$Seed = 0,
    [string]$Tag = "local_cpu"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir
try {
    $runnerArgs = @(
        "train_bio.py",
        "--data", $Data,
        "--max-steps", $Steps,
        "--seed", $Seed,
        "--tag", $Tag,
        "--device", "cpu",
        "--cpu-threads", $Threads,
        "--interop-threads", "1",
        "--token-cache", "auto"
    )
    & $Python @runnerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "train_bio.py failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
