@echo off
REM 一键打包 Windows exe（.bat 版本，双击或命令行直接运行，无执行策略限制）
REM 产物：dist\NoiseDefense.exe（单文件），config.yaml 与 sounds\ 放在同目录
setlocal
cd /d "%~dp0.."

REM 1) 环境
if not exist .venv (
    echo [1/3] 创建虚拟环境...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo [1/3] 安装依赖...
pip install -q -r requirements.txt pyinstaller

REM 2) 打包
echo [2/3] PyInstaller 打包...
pyinstaller --noconfirm --onefile --windowed --name NoiseDefense ^
  --collect-all sounddevice --collect-all miniaudio ^
  main.py
if errorlevel 1 (
    echo 打包失败，请查看上方错误信息
    pause
    exit /b 1
)

REM 3) 配置与音效放 exe 旁
echo [3/3] 复制配置与反馈音...
if not exist dist\sounds mkdir dist\sounds
copy /Y config.yaml dist\ >nul
xcopy /E /I /Y sounds\default dist\sounds\default >nul

echo.
echo 完成：dist\NoiseDefense.exe（双击运行；日志在 dist\logs\app.log）
pause
