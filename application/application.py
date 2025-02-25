#!/usr/bin/python3
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QApplication
from log_message_type import LogMessageType
from main_window import QRobotMainWindow
from robot import QRobot
import platform
import sys


class QRobotApplication(QApplication):
    log_signal = pyqtSignal(object, object)
    get_frame_signal = pyqtSignal()
    connection = None

    def __init__(self, argv):
        super().__init__(argv)
        self.robot = QRobot(self)

    def start(self, window):
        # Главное окно
        self.window = window
        self.log_signal.connect(self.window.log)
        self.log_signal.emit(f"Начало работы на {platform.uname().system}", LogMessageType.STATUS)
        self.robot.show_frame_signal.connect(self.window.show_frame)

    def stop(self):
        self.robot.stop()

    def log(self, message, type = LogMessageType.STATUS):
        self.log_signal.emit(message, type)

if __name__ == "__main__":
    app = QRobotApplication(sys.argv)
    _main_window = QRobotMainWindow(app)
    _main_window.show()
    app.start(_main_window)
    app.exec()
    app.stop()