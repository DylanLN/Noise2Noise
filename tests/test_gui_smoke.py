"""GUI 冒烟测试（offscreen）。验证窗口能构建、音量条逻辑与检测开关不崩溃。"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from config import Config
from main import Controller


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


def _make_window():
    from gui import MainWindow
    cfg = Config()
    ctrl = Controller(cfg)
    return MainWindow(ctrl, cfg), cfg


def test_window_builds(app):
    win, _ = _make_window()
    assert win.windowTitle() == "Noise Defense System"
    win.close()


def test_detector_toggle_disables(app):
    win, _ = _make_window()
    # 取消勾选"撞击" → ImpactDetector 被禁用
    win.det_checks["撞击"].setChecked(False)
    impact = next(d for d in win.controller.detectors if d.name == "Impact")
    assert not impact.enabled
    win.det_checks["撞击"].setChecked(True)
    assert impact.enabled


def test_level_bar_mapping(app):
    win, cfg = _make_window()
    from dsp import Feature
    win._on_feature(Feature(ts=1.0, rms=0.5, low_energy_ratio=0.6,
                            mid_energy_ratio=0.3, high_energy_ratio=0.1))
    assert win.bar_rms.value() >= 0
    assert win.bar_low.value() == 60
    assert win.bar_mid.value() == 30
    assert win.bar_high.value() == 10
    win.close()


def test_schedule_widget_applies(app):
    win, cfg = _make_window()
    win.win_edit.setText("22:00-01:00")
    win._on_add_window()
    assert "22:00-01:00" in cfg.active_windows
    assert win.controller.schedule.windows  # 已重新解析
    win.close()
