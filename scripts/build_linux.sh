#!/usr/bin/env bash
# 一键打包 Linux 可执行文件（PyInstaller onefile）
# 产物：dist/NoiseDefense（单文件可执行），config.yaml 与 sounds/ 放在同目录
set -euo pipefail
cd "$(dirname "$0")/.."

# 1) 环境
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt pyinstaller

# 2) 打包（onedir 便于调试，产物在 dist/NoiseDefense/）
pyinstaller --noconfirm --onedir --name NoiseDefense \
  --collect-all sounddevice --collect-all miniaudio \
  main.py

# 3) 把可编辑配置与反馈音放到可执行文件旁
DIST=dist/NoiseDefense
mkdir -p "$DIST/sounds"
cp config.yaml "$DIST/"
cp -r sounds/default "$DIST/sounds/"

echo ""
echo "✔ 完成：$DIST/NoiseDefense"
echo "  运行：dist/NoiseDefense/NoiseDefense"
echo "  运行时系统依赖："
echo "    sudo apt install -y libxcb-cursor0 libxcb-icccm4 libxcb-image0 \\"
echo "      libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \\"
echo "      libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 libportaudio2"
