#!/usr/bin/python3
from PyQt6.QtCore import QThread, pyqtSignal,  pyqtSlot, QTimer
from PyQt6.QtWidgets import QApplication

from servo_controller import QServoController
from log_message_type import LogMessageType
from main_window import QRobotMainWindow
from robot import QRobot
from camera import QRobotCamera
from voice import QRobotVoice
import platform
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
        # Главное окно
        self.window = window
        self.log_signal.connect(self.window.log)
        self.log_signal.emit(f"Начало работы на {platform.uname().system}", LogMessageType.STATUS)

        # Робот
        self.robot = QRobot(self.servo_controller)

        # Камера
        self.camera = QRobotCamera()
        self.camera.frame_captured_signal.connect(self.on_frame_captured)
        self.camera.start()
        self.camera.get_frame() # Получение первого кадра

        # Голос
        self.voice_thread = QThread()
        self.voice = QRobotVoice()
        self.voice.moveToThread(self.voice_thread)
        self.voice.phrase_captured_signal.connect(self.on_phrase_captured)
        self.voice_thread.started.connect(self.voice.listen)
        self.voice_thread.start()

    def stop(self):
        self.camera.stop()

    @pyqtSlot(object)
    def on_frame_captured(self, frame):
        frame, data = self.robot.process_frame(frame)
        self.show_frame_signal.emit(frame)
        # Получение следующего кадра
        QTimer.singleShot(10, self.camera.get_frame)

    @pyqtSlot(object)
    def on_phrase_captured(self, phrase):
        self.log_signal.emit(f"Услышал фразу: {phrase}", LogMessageType.STATUS)

if __name__ == "__main__":
    app = QRobotApplication(sys.argv)
    _main_window = QRobotMainWindow(app)
    _main_window.show()
    app.start(_main_window)
    app.exec()
    app.stop()