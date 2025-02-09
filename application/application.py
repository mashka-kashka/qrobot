#!/usr/bin/python3
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QTimer
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication
from PyQt6.QtMultimedia import QCamera, QImageCapture, QMediaCaptureSession
from servo_controller import QServoController
from log_message_type import LogMessageType
from main_window import QRobotMainWindow
from robot import QRobot
import platform
import toml
import sys


class QRobotApplication(QApplication):
    log_signal = pyqtSignal(object, object)
    show_frame_signal = pyqtSignal(object)
    get_frame_signal = pyqtSignal()
    connection = None

    def __init__(self, argv):
        super().__init__(argv)
        self.servo_controller = QServoController()

    def start(self, window):

        with open('config.toml', 'r') as f:
            self.config = toml.load(f)
            self.camera_index = self.config["camera"]["index"]
            self.frame_format = self.config["camera"]["format"]
            self.frame_size = (self.config["camera"]["width"], self.config["camera"]["height"])

        # Главное окно
        self.window = window
        self.log_signal.connect(self.window.log)
        self.log_signal.emit(f"Начало работы на {platform.uname().system}", LogMessageType.STATUS)

        # Робот
        self.robot = QRobot(self.servo_controller)

        # Камера
        self.capture_session = QMediaCaptureSession()
        self.camera = QCamera()
        self.capture_session.setCamera(self.camera)
        self.image_capture = QImageCapture(self.camera)
        self.capture_session.setImageCapture(self.image_capture)
        self.image_capture.imageCaptured.connect(self.on_image_captured)
        self.camera.start()

        # Получение первого кадра
        self.image_capture.capture()

    def stop(self):
        self.camera.stop()

    @pyqtSlot(int, QImage)
    def on_image_captured(self, id, image):
        self.image_capture.imageExposed.emit(id)
        frame, data = self.robot.process_frame(image)
        self.show_frame_signal.emit(frame)

        # Получение следующего кадра
        QTimer.singleShot(10, self.image_capture.capture)

if __name__ == "__main__":
    app = QRobotApplication(sys.argv)
    _main_window = QRobotMainWindow(app)
    _main_window.show()
    app.start(_main_window)
    app.exec()
    app.stop()