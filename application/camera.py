from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import cv2
import platform
import toml


class Camera(QObject):
    activate_robot_signal = pyqtSignal(bool)
    activate_computer_signal = pyqtSignal(bool)
    frame_captured_signal = pyqtSignal(object)
    picam2 = None
    cap = None
    running = False

    def __init__(self, camera_index=0):
        super().__init__()
        with open('config.toml', 'r') as f:
            self.config = toml.load(f)
        self.camera_index = camera_index

    @pyqtSlot()
    def start(self):
        self.running = True
        if platform.uname().node == "raspberrypi":
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(
                main={"format": 'RGB888', "size": (self.config["camera"]["width"], self.config["camera"]["height"])}))
            self.picam2.start() # запускаем камеру
            self.activate_robot_signal.emit(False)
            self.get_frame()
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.get_frame()
            else:
                self.running = False
                self.activate_computer_signal.emit(False)

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