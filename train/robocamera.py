from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread
import platform
import cv2

class RoboCamera(QObject):
    started_signal = pyqtSignal()
    failed_signal = pyqtSignal()
    frame_captured_signal = pyqtSignal(object)
    picam2 = None
    cap = None
    running = False

    def __init__(self, config):
        super().__init__()
        self.config = config

    @pyqtSlot()
    def start(self):
        if platform.uname().node == "raspberrypi":
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(
                main={"format": self.config["camera"]["format"],
                      "size": (self.config["camera"]["width"],
                               self.config["camera"]["height"])}))
            self.picam2.start() # запускаем камеру
            self.running = True
            self.started_signal.emit()
            self.get_frame()
        else:
            self.cap = cv2.VideoCapture(self.config["camera"]["index"])
            if self.cap.isOpened():
                self.running = True
                self.started_signal.emit()
                self.get_frame()
            else:
                self.running = False
                self.failed_signal.emit()

    def stop(self):
        self.running = False
        if self.picam2:
            self.picam2.stop()
        if self.cap:
            self.cap.release()

    @pyqtSlot()
    def get_frame(self):
        if not self.running:
            return
        if self.picam2:
            _frame = self.picam2.capture_array()
            self.frame_captured_signal.emit(_frame)
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.frame_captured_signal.emit(frame)