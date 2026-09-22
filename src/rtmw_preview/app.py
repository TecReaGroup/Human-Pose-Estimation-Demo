"""Threaded camera inference and PySide6 preview."""

import logging
import sys
import threading
import time
import tomllib
from collections import deque

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPainter
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox, QWidget

from rtmw_preview.runtime import PERF, ROOT, configure_logging

LOGGER = logging.getLogger("preview")
CAMERA_TIMEOUT = 5.0


class CameraWorker(QThread):
    """Publish only the newest completed frame without queuing GUI signals."""

    status = Signal(str)
    failed = Signal(str)

    def __init__(self, settings: dict) -> None:
        super().__init__()
        self.settings = settings
        self.lock = threading.Lock()
        self.latest: tuple[QImage, float] | None = None

    def take_frame(self) -> tuple[QImage, float] | None:
        """Consume the latest image once."""
        with self.lock:
            latest, self.latest = self.latest, None
        return latest

    def run(self) -> None:
        """Initialize GPU sessions, acquire frames and render pose overlays."""
        camera = None
        try:
            from rtmw_preview.pose import load_pose

            sys.path.insert(0, str(ROOT))
            from device.UsbCamera import UsbCamera

            inference = self.settings["inference"]
            self.status.emit(f"正在加载 RTMW balanced / {inference['engine']}…")
            pose = load_pose(inference)
            if self.isInterruptionRequested():
                return
            self.status.emit("正在打开相机…")
            camera = UsbCamera(self.settings["camera"])
            camera.open()
            LOGGER.info("Camera opened: index=%s backend=%s", camera.cameraId, camera.backend)
            completed = deque(maxlen=30)
            last_capture = time.perf_counter()
            last_report = last_capture
            self.status.emit(f"RTMW balanced · {inference['engine']}")
            while not self.isInterruptionRequested():
                _, frame = camera.getFrame(timeout=0.1)
                if frame is None:
                    if time.perf_counter() - last_capture > CAMERA_TIMEOUT:
                        raise RuntimeError("相机超过 5 秒没有返回画面，请检查连接或相机占用。")
                    continue
                while True:
                    _, newer = camera.getFrame(timeout=0)
                    if newer is None:
                        break
                    frame = newer
                last_capture = time.perf_counter()
                rendered = pose.render(frame)
                height, width = rendered.shape[:2]
                image = QImage(
                    rendered.data, width, height, rendered.strides[0], QImage.Format.Format_BGR888
                ).copy()
                now = time.perf_counter()
                completed.append(now)
                fps = (len(completed) - 1) / (now - completed[0]) if len(completed) > 1 else 0.0
                with self.lock:
                    self.latest = image, fps
                if now - last_report >= 10:
                    LOGGER.log(PERF, "Camera + inference + rendering FPS: %.1f", fps)
                    last_report = now
        except Exception as exc:
            LOGGER.exception("Preview stopped")
            self.failed.emit(str(exc))
        finally:
            if camera is not None:
                camera.stopThread()
                LOGGER.info("Camera released")


class Preview(QWidget):
    """Paint an aspect-preserving camera image and a fixed top-right FPS label."""

    def __init__(self) -> None:
        super().__init__()
        self.image = QImage()
        self.fps_label = QLabel("FPS  0.0", self)
        self.fps_label.setStyleSheet(
            "background: rgba(0,0,0,180); color: #62ef9b; padding: 10px;"
            "font: bold 20px 'Consolas'; border-radius: 6px;"
        )
        self.fps_label.setFixedSize(160, 48)

    def paintEvent(self, event) -> None:
        """Draw the current frame centered on a black canvas."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if not self.image.isNull():
            size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            left = (self.width() - size.width()) // 2
            top = (self.height() - size.height()) // 2
            painter.drawImage(left, top, self.image.scaled(size, Qt.AspectRatioMode.KeepAspectRatio))

    def resizeEvent(self, event) -> None:
        """Keep the FPS overlay anchored to the upper-right corner."""
        self.fps_label.move(max(0, self.width() - 176), 16)
        super().resizeEvent(event)


class PreviewWindow(QMainWindow):
    """Coordinate the responsive UI and asynchronous worker shutdown."""

    def __init__(self, settings: dict) -> None:
        super().__init__()
        self.setWindowTitle(f"RTMW balanced | {settings['inference']['engine']}")
        self.resize(1280, 760)
        self.preview = Preview()
        self.setCentralWidget(self.preview)
        self.closing = False
        self.last_display = time.perf_counter()
        self.worker = CameraWorker(settings)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.failed.connect(self.show_failure)
        self.worker.finished.connect(self.worker_finished)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(16)
        self.worker.start()

    def refresh(self) -> None:
        """Display completed frames and clear stale FPS readings."""
        latest = self.worker.take_frame()
        if latest is not None:
            self.preview.image, fps = latest
            self.preview.fps_label.setText(f"FPS  {fps:.1f}")
            self.preview.update()
            self.last_display = time.perf_counter()
        elif time.perf_counter() - self.last_display > 1:
            self.preview.fps_label.setText("FPS  0.0")

    def show_failure(self, message: str) -> None:
        """Show worker failures in the GUI thread."""
        self.statusBar().showMessage(message)
        if not self.closing:
            QMessageBox.critical(self, "预览失败", message)

    def worker_finished(self) -> None:
        """Finish a deferred close after camera resources are released."""
        if self.closing:
            self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Request cooperative shutdown without blocking the GUI thread."""
        if self.worker.isRunning():
            self.closing = True
            self.worker.requestInterruption()
            self.statusBar().showMessage("正在释放资源；引擎构建期间需等待当前操作完成…")
            event.ignore()
        else:
            self.timer.stop()
            event.accept()


def main() -> int:
    """Start the desktop camera preview using project configuration."""
    configure_logging()
    application = QApplication(sys.argv)
    try:
        with (ROOT / "config" / "config.toml").open("rb") as stream:
            settings = tomllib.load(stream)
        window = PreviewWindow(settings)
    except Exception as exc:
        LOGGER.exception("Application startup failed")
        QMessageBox.critical(None, "启动失败", str(exc))
        return 1
    window.show()
    LOGGER.info("Starting RTMW balanced preview")
    return application.exec()
