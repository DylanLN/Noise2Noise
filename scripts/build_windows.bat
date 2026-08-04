@echo off
REM One-click Windows build (ASCII only: avoids codepage issues in cmd.exe)
REM Output: dist\NoiseDefense.exe + config.yaml + sounds\ beside it
setlocal
cd /d "%~dp0.."

if not exist .venv (
    echo [1/3] Creating virtualenv...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo [1/3] Installing dependencies...
pip install -q -r requirements.txt pyinstaller

echo [2/3] PyInstaller build (this may take a few minutes)...
pyinstaller --noconfirm --onefile --windowed --name NoiseDefense --collect-all sounddevice --collect-all miniaudio main.py
if errorlevel 1 (
    echo BUILD FAILED. See errors above.
    pause
    exit /b 1
)

echo [3/3] Copying config.yaml and sounds...
if not exist dist\sounds mkdir dist\sounds
copy /Y config.yaml dist\ >nul
xcopy /E /I /Y sounds\default dist\sounds\default >nul

echo.
echo DONE: dist\NoiseDefense.exe  (double-click to run)
pause
