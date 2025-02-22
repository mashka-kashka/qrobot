from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtMultimedia import QCamera, QImageCapture, QMediaCaptureSession
import numpy as np
import platform
import toml


class QRobotCamera(QObject):
    frame_captured_signal = pyqtSignal(object)
    picam2 = None
    camera = None
    running = False

    def __init__(self):
        super().__init__()

        self.format = 'RGB888'
        self.size = (1024, 768)
        with open('config.toml', 'r') as f:
            self.config = toml.load(f)
            self.format = self.config["camera"]["format"]
            self.size = (self.config["camera"]["width"], self.config["camera"]["height"])
        try:
            if platform.uname().node == "raspberrypi":
                from picamera2 import Picamera2
                self.picam2 = Picamera2()
                self.picam2.configure(self.picam2.create_preview_configuration(
                    main={"format": self.format, "size": self.size}))
            else:
                self.capture_session = QMediaCaptureSession()
                self.camera = QCamera()
                self.capture_session.setCamera(self.camera)
                self.image_capture = QImageCapture(self.camera)
                self.capture_session.setImageCapture(self.image_capture)
                self.image_capture.imageCaptured.connect(self.on_image_captured)
        except Exception as e:
            print(f'Ошибка камеры: {str(e)}')

    @pyqtSlot()
    def start(self):
        self.running = True
        if self.picam2:
            self.picam2.start()
        if self.camera:
            self.camera.start()
        self.get_frame() # Получение первого кадра

    def stop(self):
        self.running = False
        if self.picam2:
            self.picam2.stop()
        if self.camera:
            self.camera.stop()

    @pyqtSlot()
    def get_frame(self):
        if not self.running:
            return

        try:
            if self.picam2:
                _frame = self.picam2.capture_array()
                self.frame_captured_signal.emit(_frame)
            if self.image_capture:
                self.image_capture.capture()
        except Exception:
            pass

    @pyqtSlot(int, QImage)
    def on_image_captured(self, id, image):
        _frame = image.convertToFormat(QImage.Format.Format_RGB888)

        width = image.width()
        height = image.height()

        ptr = _frame.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        self.frame_captured_signal.emit(arr)


