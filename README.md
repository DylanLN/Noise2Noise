# Noise Defense System

Windows 家庭噪声事件检测与自动响应系统（"以噪治噪"）。监听楼板传来的脚步声/拍球/拖家具/撞击/蹦跳/摔门，在允许的时间窗口内自动播放反馈音。纯规则、无 AI、无联网、本地运行。

> 基于 `docs/设计文档.md`（V5.1）实现，V1 为精简版。

## 硬件假设

- 输入：普通有线/USB 麦克风（含笔记本内置、耳机麦）。V1 暂不做蓝牙专项适配，如用蓝牙设备在 `config.yaml` 里把 `input_type` 手动设为 `bluetooth`。
- 输出：任意音响设备。**输入输出尽量分离**，避免同设备双工降质。

## 开发环境（Ubuntu）

PyQt6 需要 xcb 平台插件、sounddevice 需要 PortAudio，先装系统依赖：

```bash
sudo apt install -y \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 \
  libxcb-xkb1 libxkbcommon-x11-0 libportaudio2
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

> 注意：本机 shell 的 `PYTHONPATH` 若指向 ROS 路径，pytest 会误加载 ROS 插件，用
> `env -u PYTHONPATH .venv/bin/python -m pytest tests/` 运行测试。

## 运行

```bash
. .venv/bin/activate
python main.py                     # 启动 GUI（无显示环境自动退回控制台模式）
```

首次启动约 10 秒会进行本底噪声标定（期间请保持安静），之后进入监听。

### 生成默认反馈音

```bash
. .venv/bin/activate
python tools/gen_sound.py          # 生成 sounds/default/boom.wav
```

也可以直接把任意 `wav / mp3 / flac` 放进 `sounds/` 目录，系统随机挑选播放。

## 配置（config.yaml）

| 键 | 说明 |
|---|---|
| `sensitivity` | 容忍度，越大越不敏感（阈值 = baseline + sensitivity×std） |
| `cooldown` | 每次响应后的冷却秒数 |
| `confirm_count` / `confirm_window_sec` | 滑动窗口内 Episode 数达标才触发 |
| `active_windows` | 允许自动响应的时间段，支持跨天（如 `22:00-01:00`） |
| `max_responses_per_window` | 每个活跃时段内最多自动响应次数 |
| `fresh_episode_gap_sec` | 距上次响应需静默该时长才允许再次响应（防"以噪治噪"升级） |
| `volume` / `sounds_dir` | 播放音量与反馈音目录 |

## 测试

```bash
env -u PYTHONPATH .venv/bin/python -m pytest tests/ -v
```

覆盖：滤波、特征提取、本底标定、6 类检测规则、滞回状态机、Episode 语义、仲裁、时间窗口、Mute Gate、Controller 集成、GUI 冒烟。

## 打包成软件

依赖 PyInstaller，产物里会带上 `config.yaml` 和 `sounds/`（放在可执行文件旁边，可编辑）。

### Windows（在 Windows 上执行）

```bat
scripts\build_windows.bat
:: 或：powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
:: 产物：dist\NoiseDefense.exe —— 双击运行；日志在 dist\logs\app.log
```

### Ubuntu（在本机执行）

```bash
./scripts/build_linux.sh
# 产物：dist/NoiseDefense/NoiseDefense —— 运行前确保 sudo apt install libportaudio2
```

> 打包后配置/音效路径自动解析到可执行文件所在目录（见 `paths.py`），与源码位置无关。

## 待办（V2）

- 蓝牙 HFP 适配（采样率协商、延迟自测、重连）
- 离线阈值标定工具（录样本 → 扫阈值 → 写回配置）
- 实时频谱面板、Windows 打包（PyInstaller）
