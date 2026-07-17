param(
    [switch]$AllowCpu,
    [switch]$Recreate,
    [int]$Timesteps = 300000,
    [string[]]$Archs = @("transformer_gnn"),
    [int[]]$Seeds = @(0)
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "=== Swarm-SAR GPU training ===" -ForegroundColor Cyan

function Require-Python311 {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "Python was not found."
    }
    $version = & python --version 2>&1
    if ($LASTEXITCODE -ne 0 -or $version -notmatch "Python 3\.11\.") {
        throw "Python 3.11 is required. Found: $version"
    }
    Write-Host "Using $version"
}

Require-Python311

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} elseif (-not $AllowCpu) {
    throw "nvidia-smi was not found. Re-run with -AllowCpu to permit CPU training."
}

if ($Recreate -and (Test-Path ".venv")) {
    Write-Host "Removing the existing .venv as requested"
    Remove-Item -LiteralPath ".venv" -Recurse -Force
}

if (-not (Test-Path ".venv")) {
    Write-Host "[1/5] Creating Python 3.11 virtual environment"
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
$venvVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($venvVersion -ne "3.11") {
    throw ".venv is Python $venvVersion. Re-run with -Recreate to rebuild it with Python 3.11."
}

Write-Host "[2/5] Upgrading pip"
python -m pip install --upgrade pip

Write-Host "[3/5] Installing CUDA PyTorch and project dependencies"
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
if (Test-Path "requirements-dev.txt") {
    pip install -r requirements-dev.txt
}
pip install -e ".[rl,viz]"

Write-Host "[4/5] Verifying CUDA runtime"
$verifyArgs = @("scripts\verify_gpu.py")
if (-not $AllowCpu) {
    $verifyArgs += "--require-cuda"
}
python @verifyArgs
if ($LASTEXITCODE -ne 0) {
    throw "GPU verification failed. Re-run with -AllowCpu only if CPU training is intentional."
}

Write-Host "[5/5] Smoke test, then training"
python scripts\train_and_report.py --smoke
$trainArgs = @(
    "scripts\train_and_report.py",
    "--config", "configs\training\mappo_easy.yaml",
    "--archs"
) + $Archs + @("--seeds") + $Seeds + @("--timesteps", $Timesteps)
python @trainArgs

Write-Host "DONE. Figures in results\trained\figures ; report in results\trained\report.json" -ForegroundColor Green
