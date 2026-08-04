"""PyQt6 精简界面：设备选择 / 检测类型 / 容忍度 / 时间窗口 / 四根音量条 / 日志。
跨线程更新通过 pyqtSignal（DSP 线程 emit → 主线程队列处理）。"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QMainWindow, QPlainTextEdit, QProgressBar, QPushButton,
                             QSlider, QVBoxLayout, QWidget)

from audio import list_input_devices, list_output_devices
from config import Config
from main import Controller

# GUI 中文标签 → Detector 名
DETECTOR_MAP = {"跑步": "Footstep", "拍球": "Ball", "拖家具": "Chair",
                "撞击": "Impact", "蹦跳": "Jump", "摔门": "Door"}


class MainWindow(QMainWindow):
    feature_signal = pyqtSignal(object)      # DSP 线程 → 主线程，携带 Feature
    log_signal = pyqtSignal(str)

    def __init__(self, controller: Controller, cfg: Config):
        super().__init__()
        self.controller = controller
        self.cfg = cfg
        self._last_ui = 0.0                  # 音量条节流时间戳
        self.setWindowTitle("Noise Defense System")
        self.resize(520, 720)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # 设备
        dev = QFormLayout()
        self.in_combo = QComboBox()
        self.out_combo = QComboBox()
        dev.addRow("输入设备", self.in_combo)
        dev.addRow("输出设备", self.out_combo)
        root.addLayout(dev)

        # 检测类型
        det_box = QGroupBox("检测类型")
        det_lay = QVBoxLayout(det_box)
        self.det_checks: dict[str, QCheckBox] = {}
        for label in DETECTOR_MAP:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(self._on_detector_toggle)
            det_lay.addWidget(cb)
            self.det_checks[label] = cb
        root.addWidget(det_box)

        # 容忍度 / 冷却 / 触发次数（滑条右侧显示当前值）
        tol = QFormLayout()

        self.sensitivity = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity.setRange(1, 20)
        self.sensitivity.setValue(int(cfg.sensitivity))
        self.sensitivity.valueChanged.connect(self._on_sensitivity)
        self.sens_label = QLabel(f"{self.sensitivity.value()}/20")
        row = QHBoxLayout(); row.addWidget(self.sensitivity); row.addWidget(self.sens_label)
        tol.addRow("容忍度", row)

        self.cooldown = QSlider(Qt.Orientation.Horizontal)
        self.cooldown.setRange(1, 60)
        self.cooldown.setValue(int(cfg.cooldown))
        self.cooldown.valueChanged.connect(self._on_cooldown)
        self.cool_label = QLabel(f"{self.cooldown.value()}/60 秒")
        row = QHBoxLayout(); row.addWidget(self.cooldown); row.addWidget(self.cool_label)
        tol.addRow("冷却", row)

        self.confirm = QSlider(Qt.Orientation.Horizontal)
        self.confirm.setRange(1, 5)
        self.confirm.setValue(cfg.confirm_count)
        self.confirm.valueChanged.connect(self._on_confirm)
        self.confirm_label = QLabel(f"{self.confirm.value()}/5 次触发")
        row = QHBoxLayout(); row.addWidget(self.confirm); row.addWidget(self.confirm_label)
        tol.addRow("触发次数", row)

        root.addLayout(tol)

        # 时间窗口
        win_box = QGroupBox("自动响应时间窗口")
        win_lay = QVBoxLayout(win_box)
        self.win_list = QListWidget()
        self.win_list.addItems(cfg.active_windows)
        self.win_edit = QLineEdit()
        self.win_edit.setPlaceholderText("HH:MM-HH:MM，如 22:00-01:00（跨天）")
        self.win_add = QPushButton("添加")
        self.win_add.clicked.connect(self._on_add_window)
        win_row = QHBoxLayout()
        win_row.addWidget(self.win_edit)
        win_row.addWidget(self.win_add)
        win_lay.addWidget(self.win_list)
        win_lay.addLayout(win_row)
        self.always_check = QCheckBox("始终允许响应（测试用，忽略时间窗口）")
        self.always_check.setChecked(cfg.schedule_always)
        self.always_check.toggled.connect(self._on_always)
        win_lay.addWidget(self.always_check)
        root.addWidget(win_box)

        # 回击音文件
        fb = QFormLayout()
        self.fb_path = QLineEdit(cfg.feedback_file)
        self.fb_path.setReadOnly(True)
        self.fb_path.setPlaceholderText("留空 = 从 sounds/ 随机播放")
        self.fb_browse = QPushButton("浏览…")
        self.fb_browse.clicked.connect(self._on_pick_sound)
        fb_row = QHBoxLayout()
        fb_row.addWidget(self.fb_path)
        fb_row.addWidget(self.fb_browse)
        fb.addRow("回击音文件", fb_row)
        root.addLayout(fb)

        # 音量条（代替频谱）
        lvl = QFormLayout()
        self.bar_rms = QProgressBar()
        self.bar_low = QProgressBar()
        self.bar_mid = QProgressBar()
        self.bar_high = QProgressBar()
        for bar, label in ((self.bar_rms, "当前音量"), (self.bar_low, "低频 20-150Hz"),
                           (self.bar_mid, "中频 150-1kHz"), (self.bar_high, "高频 1kHz+")):
            bar.setRange(0, 100)
            lvl.addRow(label, bar)
        root.addLayout(lvl)

        # 控制
        ctrl_row = QHBoxLayout()
        self.start_btn = QPushButton("开始监听")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        ctrl_row.addWidget(self.start_btn)
        ctrl_row.addWidget(self.stop_btn)
        root.addLayout(ctrl_row)

        self.status = QLabel("状态：停止（等待标定后进入监听）")
        root.addWidget(self.status)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        root.addWidget(self.log_view)

        # 设备下拉（此时 log_view 已就绪，OSError 可安全记日志）
        self._fill_devices()

        # 接线：DSP 线程 → 信号 → 主线程槽
        self.controller.on_feature = self.feature_signal.emit
        self.controller.on_log = self.log_signal.emit
        self.feature_signal.connect(self._on_feature)
        self.log_signal.connect(self._on_log)

    # ── 设备 ──
    def _fill_devices(self) -> None:
        for combo, fn in ((self.in_combo, list_input_devices),
                          (self.out_combo, list_output_devices)):
            combo.addItem("默认设备")
            try:
                for name in fn():
                    combo.addItem(name)
            except OSError as e:
                self._on_log(f"音频设备不可用：{e}")

    # ── 控件回调 ──
    def _on_detector_toggle(self, checked: bool) -> None:
        sender = self.sender()
        name = DETECTOR_MAP.get(sender.text())
        for d in self.controller.detectors:
            if d.name == name:
                d.enabled = checked

    def _on_sensitivity(self, v: int) -> None:
        self.cfg.sensitivity = float(v)
        self.sens_label.setText(f"{v}/20")

    def _on_cooldown(self, v: int) -> None:
        self.cfg.cooldown = float(v)
        self.cool_label.setText(f"{v}/60 秒")

    def _on_confirm(self, v: int) -> None:
        self.cfg.confirm_count = v
        self.controller.em.default_confirm_count = v
        self.confirm_label.setText(f"{v}/5 次触发")

    def _on_add_window(self) -> None:
        spec = self.win_edit.text().strip()
        if spec:
            self.win_list.addItem(spec)
            self.cfg.active_windows = self._current_windows()
            self.controller.schedule.set_windows(self.cfg.active_windows)
            self.win_edit.clear()

    def _on_always(self, checked: bool) -> None:
        self.cfg.schedule_always = checked
        self.controller.schedule.always = checked

    def _on_pick_sound(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择回击音文件", "", "音频文件 (*.wav *.mp3 *.flac)")
        if path:
            self.set_feedback_file(path)

    def set_feedback_file(self, path: str) -> None:
        self.cfg.feedback_file = path
        self.fb_path.setText(path)
        self._on_log(f"回击音设为：{path}")

    def _current_windows(self) -> list[str]:
        return [self.win_list.item(i).text() for i in range(self.win_list.count())]

    # ── 启动/停止 ──
    def start(self) -> None:
        try:
            self.controller.start()
        except OSError as e:
            self._on_log(f"启动失败：{e}")
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("状态：监听中（标定中…）")

    def stop(self) -> None:
        self.controller.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("状态：停止")

    # ── 跨线程槽 ──
    def _on_feature(self, f) -> None:
        if f.ts - self._last_ui < 0.1:       # 节流到 ~10Hz
            return
        self._last_ui = f.ts
        base = max(self.controller.baseline.baseline_rms, 1e-6)
        self.bar_rms.setValue(min(100, int(f.rms / base * 20)))
        self.bar_low.setValue(min(100, int(f.low_energy_ratio * 100)))
        self.bar_mid.setValue(min(100, int(f.mid_energy_ratio * 100)))
        self.bar_high.setValue(min(100, int(f.high_energy_ratio * 100)))
        if self.controller.baseline.calibrated:
            self.status.setText("状态：监听中")
        if not self.start_btn.isEnabled():
            pass                              # 已在监听，状态文案由上行维护

    def _on_log(self, msg: str) -> None:
        if hasattr(self, "log_view"):
            self.log_view.appendPlainText(msg)
