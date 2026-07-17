@echo off
REM ============================================================
REM  Swarm-SAR - one-click training on a Python 3.11 CUDA setup.
REM  Pass --allow-cpu to continue when CUDA is unavailable.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set ALLOW_CPU=0
set RECREATE=0
for %%A in (%*) do (
  if /I "%%~A"=="--allow-cpu" set ALLOW_CPU=1
  if /I "%%~A"=="--recreate" set RECREATE=1
)

echo(
echo === Swarm-SAR GPU training ===
py -3.11 --version >nul 2>nul
if errorlevel 1 (
  echo Python 3.11 is required. Install it and verify: py -3.11 --version
  exit /b 1
)
py -3.11 --version

where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo Detected NVIDIA GPU:
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
) else (
  if "%ALLOW_CPU%"=="0" (
    echo nvidia-smi was not found. Re-run with --allow-cpu to permit CPU training.
    exit /b 1
  )
  echo No NVIDIA GPU detected - CPU training explicitly allowed.
)

echo(
echo [1/5] Creating Python 3.11 virtual environment (.venv) ...
if "%RECREATE%"=="1" if exist ".venv" (
  echo Removing the existing .venv as requested...
  rmdir /s /q ".venv"
)
if not exist ".venv" ( py -3.11 -m venv .venv )
call .venv\Scripts\activate.bat
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
if errorlevel 1 (
  echo .venv is not Python 3.11. Rerun with --recreate to rebuild it safely.
  exit /b 1
)
python -m pip install --upgrade pip

echo(
echo [2/5] Installing PyTorch CUDA 12.1 build ...
pip install torch --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 exit /b 1

echo(
echo [3/5] Installing Swarm-SAR dependencies ...
pip install -r requirements.txt
if errorlevel 1 exit /b 1
pip install -r requirements-dev.txt
if errorlevel 1 exit /b 1
pip install -e ".[rl,viz]"
if errorlevel 1 exit /b 1

echo(
echo [4/5] Verifying GPU runtime ...
if "%ALLOW_CPU%"=="1" (
  python scripts\verify_gpu.py
) else (
  python scripts\verify_gpu.py --require-cuda
)
if errorlevel 1 exit /b 1

echo(
echo [5/5] Smoke test, then laptop-sized training ...
python scripts\train_and_report.py --smoke
if errorlevel 1 exit /b 1
python scripts\train_and_report.py --config configs\training\mappo_easy.yaml --archs transformer_gnn --seeds 0 --timesteps 300000
if errorlevel 1 exit /b 1

echo(
echo ============================================================
echo  DONE. Measured figures: results\trained\figures
echo  Report:                 results\trained\report.json
echo  TensorBoard:            tensorboard --logdir results\trained
echo ============================================================
pause
