"""Desktop GUI: paint noise trajectories on the spectrogram, repair, A/B.

Run:  python3 -m audio_repair.app [file.wav]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtCore import QIODevice
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QLabel, QMainWindow, QPlainTextEdit,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .dsp import StftConfig, StftProcessor, band_energy, reconstruction_report
from .io_wav import load_wav, save_wav
from .strokes import Stroke, combine_strokes
from .synth import crossed_chirps, hum_noise

_COLORMAP_STOPS = [
    (0.00, (0, 0, 4)),
    (0.25, (87, 16, 110)),
    (0.50, (188, 55, 84)),
    (0.75, (249, 142, 9)),
    (1.00, (252, 255, 164)),
]


def _colormap() -> np.ndarray:
    """256x3 uint8 inferno-like lookup table."""
    pos = np.array([s[0] for s in _COLORMAP_STOPS])
    rgb = np.array([s[1] for s in _COLORMAP_STOPS], dtype=np.float64)
    lut = np.empty((256, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.interp(np.linspace(0, 1, 256), pos,
                               rgb[:, ch]).astype(np.uint8)
    return lut


_LUT = _colormap()


class SpectrogramWidget(QWidget):
    """Spectrogram canvas; emits freehand strokes in (seconds, Hz)."""

    stroke_drawn = Signal(list)  # list[(t_sec, f_hz)]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(720, 360)
        self.setMouseTracking(True)
        self._image: QImage | None = None
        self._duration = 1.0
        self._fmax = 12000.0
        self._strokes: list[list[tuple[float, float]]] = []
        self._current: list[tuple[float, float]] = []
        self._cursor_sec: float | None = None

    def set_spectrogram(self, mag_db: np.ndarray, duration: float,
                        fmax: float) -> None:
        self._duration = max(duration, 1e-9)
        self._fmax = fmax
        lo, hi = -90.0, max(float(mag_db.max()), -20.0)
        norm = np.clip((mag_db - lo) / (hi - lo), 0.0, 1.0)
        idx = (norm * 255.0).astype(np.uint8).T[::-1]  # freq axis up
        rgb = np.ascontiguousarray(_LUT[idx])
        h, w = rgb.shape[:2]
        self._image = QImage(rgb.data, w, h, w * 3,
                             QImage.Format.Format_RGB888).copy()
        self.update()

    def set_strokes(self, strokes: list[Stroke]) -> None:
        self._strokes = [list(zip(s.times.tolist(), s.freqs.tolist()))
                         for s in strokes]
        self.update()

    def set_cursor(self, seconds: float | None) -> None:
        self._cursor_sec = seconds
        self.update()

    def _to_data(self, pos) -> tuple[float, float]:
        w, h = max(self.width(), 1), max(self.height(), 1)
        t = float(pos.x()) / w * self._duration
        f = (1.0 - float(pos.y()) / h) * self._fmax
        return t, max(f, 0.0)

    def _to_widget(self, t: float, f: float) -> tuple[float, float]:
        w, h = max(self.width(), 1), max(self.height(), 1)
        return t / self._duration * w, (1.0 - f / self._fmax) * h

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._current = [self._to_data(event.position())]
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._current:
            self._current.append(self._to_data(event.position()))
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._current:
            stroke, self._current = self._current, []
            self.stroke_drawn.emit(stroke)
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(12, 12, 16))
        if self._image is not None:
            painter.drawImage(self.rect(), self._image)
        pen = QPen(QColor(80, 255, 160, 220))
        pen.setWidth(2)
        painter.setPen(pen)
        for stroke in [*self._strokes, self._current]:
            pts = [self._to_widget(t, f) for t, f in stroke]
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                painter.drawLine(int(x0), int(y0), int(x1), int(y1))
        if self._cursor_sec is not None:
            x, _ = self._to_widget(self._cursor_sec, 0.0)
            painter.setPen(QPen(QColor(255, 255, 255, 180)))
            painter.drawLine(int(x), 0, int(x), self.height())
        painter.end()


class _PcmStream(QIODevice):
    """Pull-stream feeding the sink from the player's active buffer."""

    def __init__(self, player: "SyncPlayer") -> None:
        super().__init__(player)
        self._player = player

    def readData(self, maxlen: int) -> bytes:
        return self._player.read_frames(maxlen)

    def writeData(self, data: bytes) -> int:
        return -1

    def isSequential(self) -> bool:
        return True


class SyncPlayer(QObject):
    """Plays original/repaired buffers; A/B switch keeps the sample cursor,
    so both versions are heard exactly in sync (同步试听)."""

    position_changed = Signal(float)  # seconds
    finished = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._sr = 44100
        self._buffers: dict[str, np.ndarray] = {}
        self._active = "orig"
        self._cursor = 0
        self._sink: QAudioSink | None = None
        self._stream: _PcmStream | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)

    # -------------------------------------------------------------- setup
    def set_buffers(self, sr: int, orig: np.ndarray,
                    repaired: np.ndarray | None) -> None:
        self.stop()
        self._sr = sr
        self._buffers = {"orig": np.ascontiguousarray(
            np.clip(orig, -1, 1), dtype=np.float32)}
        if repaired is not None:
            self._buffers["repaired"] = np.ascontiguousarray(
                np.clip(repaired, -1, 1), dtype=np.float32)
        self._cursor = 0

    def set_active(self, which: str) -> None:
        if which in self._buffers:
            self._active = which  # cursor untouched -> stays in sync

    # ----------------------------------------------------------- playback
    def play(self) -> None:
        if not self._buffers:
            return
        if self._sink is None:
            fmt = QAudioFormat()
            fmt.setSampleRate(self._sr)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
            device = QMediaDevices.defaultAudioOutput()
            if device.isNull():
                return
            self._sink = QAudioSink(device, fmt, self)
            self._stream = _PcmStream(self)
            self._stream.open(QIODevice.OpenModeFlag.ReadOnly)
        if self._cursor >= len(self._buffers[self._active]):
            self._cursor = 0
        self._sink.start(self._stream)
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        if self._sink is not None:
            self._sink.stop()
        self._cursor = 0
        self.position_changed.emit(0.0)

    def read_frames(self, maxlen: int) -> bytes:
        buf = self._buffers.get(self._active)
        if buf is None:
            return bytes(maxlen)
        n = min(maxlen // 4, len(buf) - self._cursor)
        if n <= 0:
            return bytes(maxlen)  # silence past the end
        start = self._cursor
        self._cursor += n
        return buf[start:start + n].tobytes()

    def _tick(self) -> None:
        pos = self._cursor / float(self._sr)
        self.position_changed.emit(pos)
        buf = self._buffers.get(self._active)
        if buf is not None and self._cursor >= len(buf):
            self.stop()
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("窄带噪声修复 — STFT 轨迹擦除")
        self.proc = StftProcessor(StftConfig())
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.sr = 44100
        self.strokes: list[Stroke] = []

        self.spec_view = SpectrogramWidget()
        self.spec_view.stroke_drawn.connect(self._add_stroke)
        self.report = QPlainTextEdit(readOnly=True, maximumBlockCount=500)
        self.report.setMaximumHeight(150)

        self.player = SyncPlayer(self)
        self.player.position_changed.connect(self.spec_view.set_cursor)

        # ---- controls
        self.btn_open = QPushButton("打开 WAV…")
        self.btn_open.clicked.connect(self._open_wav)
        self.combo_demo = QComboBox()
        self.combo_demo.addItems(["交叉扫频+瞬态", "谐波哼声"])
        self.btn_demo = QPushButton("生成示例")
        self.btn_demo.clicked.connect(self._make_demo)
        self.spin_radius = QDoubleSpinBox(minimum=10.0, maximum=2000.0,
                                          value=150.0, suffix=" Hz")
        self.spin_depth = QDoubleSpinBox(minimum=0.05, maximum=1.0,
                                         value=1.0, singleStep=0.05)
        self.btn_undo = QPushButton("撤销笔划")
        self.btn_undo.clicked.connect(self._undo_stroke)
        self.btn_clear = QPushButton("清空笔划")
        self.btn_clear.clicked.connect(self._clear_strokes)
        self.chk_bypass = QCheckBox("旁路(校验重构)")
        self.btn_process = QPushButton("处理")
        self.btn_process.clicked.connect(self._process)
        self.combo_ab = QComboBox()
        self.combo_ab.addItems(["原音", "修复后"])
        self.combo_ab.currentIndexChanged.connect(
            lambda i: self.player.set_active("repaired" if i else "orig"))
        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self.player.play)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.player.stop)
        self.btn_save = QPushButton("导出修复…")
        self.btn_save.clicked.connect(self._save)

        bar = QHBoxLayout()
        for w in (self.btn_open, self.combo_demo, self.btn_demo,
                  QLabel("笔刷半径"), self.spin_radius,
                  QLabel("擦除深度"), self.spin_depth,
                  self.btn_undo, self.btn_clear, self.chk_bypass,
                  self.btn_process, self.combo_ab,
                  self.btn_play, self.btn_stop, self.btn_save):
            bar.addWidget(w)
        bar.addStretch(1)

        root = QVBoxLayout()
        root.addLayout(bar)
        root.addWidget(self.spec_view, stretch=1)
        root.addWidget(self.report)
        host = QWidget()
        host.setLayout(root)
        self.setCentralWidget(host)
        self._log("就绪：打开 WAV 或生成示例，然后在频谱图上圈出噪声轨迹。")

    # ------------------------------------------------------------- helpers
    def _log(self, text: str) -> None:
        self.report.appendPlainText(text)

    def _set_signal(self, x: np.ndarray, sr: int) -> None:
        self.x, self.y = x, None
        self.sr = sr
        self.proc = StftProcessor(StftConfig(sample_rate=sr))
        self.strokes.clear()
        self.spec_view.set_strokes([])
        spec = self.proc.stft(x)
        mag_db = 20.0 * np.log10(np.abs(spec).T + 1e-12).T
        fmax = min(sr / 2.0, 12000.0)
        self.spec_view.set_spectrogram(mag_db, len(x) / sr, fmax)
        self.player.set_buffers(sr, x, None)
        self.combo_ab.setCurrentIndex(0)
        self._log(f"载入信号：{len(x)/sr:.2f} s @ {sr} Hz")

    # -------------------------------------------------------------- slots
    def _open_wav(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开 WAV", "", "WAV files (*.wav)")
        if path:
            self._set_signal(*load_wav(path))

    def _make_demo(self) -> None:
        if self.combo_demo.currentIndex() == 0:
            fix = crossed_chirps()
        else:
            fix = hum_noise()
        self._set_signal(fix["mix"], fix["sr"])

    def _add_stroke(self, points: list) -> None:
        self.strokes.append(Stroke.from_points(
            points, radius_hz=self.spin_radius.value(),
            depth=self.spin_depth.value()))
        self.spec_view.set_strokes(self.strokes)
        self._log(f"笔划 #{len(self.strokes)}：{len(points)} 点，"
                  f"半径 {self.spin_radius.value():.0f} Hz")

    def _undo_stroke(self) -> None:
        if self.strokes:
            self.strokes.pop()
            self.spec_view.set_strokes(self.strokes)

    def _clear_strokes(self) -> None:
        self.strokes.clear()
        self.spec_view.set_strokes(self.strokes)

    def _process(self) -> None:
        if self.x is None:
            self._log("请先载入信号。")
            return
        cfg = self.proc.config
        spec = self.proc.stft(self.x)
        n_frames, n_bins = spec.shape
        if self.chk_bypass.isChecked() or not self.strokes:
            self.y, _ = self.proc.process(self.x, None)
            rep = reconstruction_report(self.x, self.y)
            self._log("旁路重构："
                      f"max_err={rep['max_abs_err']:.3e}  "
                      f"rmse={rep['rmse']:.3e}  "
                      f"snr={rep['snr_db']:.1f} dB  "
                      f"length_match={rep['length_match']}")
        else:
            mask = combine_strokes(self.strokes, n_frames, n_bins,
                                   cfg.frame_sec, cfg.bin_hz)
            self.y, _ = self.proc.process(self.x, mask)
            after = self.proc.stft(self.y)
            sel = band_energy(after, mask) / max(band_energy(spec, mask),
                                                 1e-30)
            rest = 1.0 - mask
            unsel = band_energy(after, rest) / max(band_energy(spec, rest),
                                                   1e-30)
            self._log(f"已处理 {len(self.strokes)} 条笔划："
                      f"选中区域能量 {10*np.log10(max(sel,1e-30)):+.1f} dB，"
                      f"未选区域能量 {10*np.log10(max(unsel,1e-30)):+.2f} dB")
        self.player.set_buffers(self.sr, self.x, self.y)

    def _save(self) -> None:
        if self.y is None:
            self._log("尚未处理，无可导出结果。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出修复", "repaired.wav", "WAV files (*.wav)")
        if path:
            save_wav(path, self.y, self.sr)
            self._log(f"已导出 {path}")


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1000, 640)
    win.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        win._set_signal(*load_wav(sys.argv[1]))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
