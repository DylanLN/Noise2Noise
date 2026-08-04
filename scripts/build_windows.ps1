# 一键打包 Windows exe（PyInstaller，无控制台窗口）
# 产物：dist\NoiseDefense.exe（单文件），config.yaml 与 sounds\ 放在同目录
# 用法：在 PowerShell 中运行  ./scripts/build_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

# 1) 环境
if (-not (Test-Path .venv)) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
pip install -q -r requirements.txt pyinstaller

# 2) 打包
pyinstaller --noconfirm --onefile --windowed --name NoiseDefense `
  --collect-all sounddevice --collect-all miniaudio `
  main.py

# 3) 把可编辑配置与反馈音放到 exe 旁
New-Item -ItemType Directory -Force dist\sounds | Out-Null
Copy-Item config.yaml dist\
Copy-Item -Recurse sounds\default dist\sounds\

Write-Host ""
Write-Host "完成：dist\NoiseDefense.exe（双击运行）"
Write-Host "配置/日志在 exe 同目录：config.yaml、logs\app.log"
