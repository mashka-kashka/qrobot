#!/usr/bin/python3
from PyQt6.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication, QSlider
from log_message_type import LogMessageType
from main_window import QRobotMainWindow
from server import QRobotServer
from client import QRobotClient
from camera import Camera
from robot import QRobot
import platform
import socket
import toml
import time
import sys


class QRobotApplication(QApplication):
    log_signal = pyqtSignal(object, object)
    send_frame_signal = pyqtSignal(object)
    send_data_signal = pyqtSignal(object)
    show_frame_signal = pyqtSignal(object)
    get_frame_signal = pyqtSignal()
    connection = None

    def __init__(self, argv):
        super().__init__(argv)
        self.sent_frames = []
        self.received_frames = []
        self.fsps = 0
        self.frps = 0

    def start(self, window):
        # Главное окно
        self.window = window
        self.log_signal.connect(self.window.log)
        self.log_signal.emit(f"Начало работы на {platform.uname().system}", LogMessageType.STATUS)

        # Робот
        self.robot = QRobot()
        self.frame_buffer = []

        # Камера
        self.camera_thread = QThread()
        self.camera = Camera()
        self.camera.moveToThread(self.camera_thread)
        self.camera.activate_robot_signal.connect(self.window.activate_robot)
        self.camera.activate_computer_signal.connect(self.window.activate_computer)
        self.camera.frame_captured_signal.connect(self.on_frame_captured)
        self.get_frame_signal.connect(self.camera.get_frame)
        self.camera_thread.started.connect(self.camera.start)

        # Сервер
        self.server = QRobotServer(self.window)
        self.server.stop_signal.connect(self.stop_server)
        self.window.start_server_signal.connect(self.start_server)
        self.window.stop_server_signal.connect(self.stop_server)
        self.window.reset_server_signal.connect(self.reset_server)

        # Клиент
        self.client = QRobotClient(self.window)
        self.client.stop_signal.connect(self.stop_client)
        self.window.start_client_signal.connect(self.start_client)
        self.window.stop_client_signal.connect(self.stop_client)
        self.window.reset_client_signal.connect(self.reset_client)

        is_on_raspberry = platform.uname().node == "raspberrypi"
        if is_on_raspberry:
            self.camera_thread.start()
        else:
            server_exists = False
            with open('config.toml', 'r') as f:
                self.config = toml.load(f)
                _video_port = int(self.config["network"]["video_port"])
                _host = self.config["network"]["host"]

                # Есть ли в сети сервер?
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_exists = sock.connect_ex((_host, _video_port)) == 0
                sock.close()

            if server_exists:
                if not is_on_raspberry:
                    self.camera_thread.start()

                self.window.activate_robot(False)
            else:
                self.window.activate_computer(False)

    def stop(self):
        self.stop_camera()
        self.stop_server()
        self.stop_client()

    def stop_camera(self):
        if self.camera_thread.isRunning():
            self.camera.stop()
            self.camera_thread.quit()
            self.camera_thread.wait()

    def is_server(self):
        return self.server.is_running()

    def start_server(self):
        self.server.start()

    def stop_server(self):
        if self.window.ui.actionActivateRobot.isChecked(): # При старте произошла ошибка
            self.window.activate_robot(True)
        self.server.stop()

    def reset_server(self):
        self.server.reset()

    def start_client(self):
        if self.is_server():
            self.server.stop()
        self.client.start()

    def stop_client(self):
        if self.window.ui.actionActivateComputer.isChecked(): # При старте произошла ошибка
            self.window.activate_computer(True)
        self.client.stop()

    def reset_client(self):
        self.client.reset()

    @pyqtSlot(object)
    def on_frame_captured(self, frame):
        if self.connection: # Есть подключение
            self.sent_frames.append(time.time_ns())
            if len(self.sent_frames) > 10:
                self.sent_frames.pop(0)
            if len(self.sent_frames) == 10:
                self.fsps = (self.sent_frames[-1] - self.sent_frames[0]) #// 10000000
                #print(f"Количество отправляемых кадров в секунду: {self.fsps}")
            self.send_frame_signal.emit(frame)
            #self.log_signal.emit(f"Отправлен кадр на обработку", LogMessageType.WARNING)
        else: # Самостоятельный расчёт
            frame, data = self.robot.process_frame(frame)
            self.show_frame_signal.emit(frame)
        self.get_frame_signal.emit()

    @pyqtSlot(object)
    def on_frame_received(self, frame):
        # Получен кадр по сети
        if self.connection.is_server: # Мы на сервере - нужно выполнить расчёт
            processed_frame, data = self.robot.process_frame(frame)
            # Активна вкладка "Сервоприводы"
            if self.window.ui.tabServos.isVisible():
                servos = {}
                for widget in self.window.ui.gl_servos.children():
                    if isinstance(widget, QSlider):
                        channel = widget.property("channel")
                        servos[channel] = widget.value()

            # Отображение обработанного кадра на сервере
            self.show_frame_signal.emit(processed_frame)

            # Отправка кадра и данных клиенту
            self.send_frame_signal.emit(processed_frame)
            self.send_data_signal.emit(data)

        else: # Мы на клиенте
            # Отображение полученного кадра
            self.show_frame_signal.emit(frame)

            self.received_frames.append(time.time_ns())
            if len(self.received_frames) > 10:
                self.received_frames.pop(0)
            if len(self.received_frames) == 10:
                self.frps = (self.received_frames[-1] - self.received_frames[0]) #// 10000000
                #print(f"Количество получаемых кадров в секунду: {self.frps} {self.fsps // self.frps}")

        self.processEvents()

    @pyqtSlot(object)
    def on_video_client_connected(self, connection):
        self.connection = connection
        connection.frame_received_signal.connect(self.on_frame_received)
        self.send_frame_signal.connect(connection.send_frame)
        self.log_signal.emit(f"Ожидание видео для обработки вычислений", LogMessageType.WARNING)

    @pyqtSlot(object)
    def on_video_connected_to_server(self, connection):
        self.connection = connection
        connection.frame_received_signal.connect(self.on_frame_received)
        self.send_frame_signal.connect(connection.send_frame)
        self.log_signal.emit(f"Видео передаётся для обработки на внешний компьютер", LogMessageType.WARNING)

    @pyqtSlot(object)
    def on_video_client_disconnected(self, connection):
        self.connection = None
        connection.frame_received_signal.disconnect(self.on_frame_received)
        self.send_frame_signal.disconnect(connection.send_frame)

    @pyqtSlot(object)
    def on_video_disconnected_from_server(self, connection):
        self.connection = None
        connection.frame_received_signal.disconnect(self.on_frame_received)
        self.send_frame_signal.disconnect(connection.send_frame)
        self.window.activate_robot(True)

    @pyqtSlot(object)
    def on_data_client_connected(self, connection):
        self.connection = connection
        connection.data_received_signal.connect(self.on_data_received)
        self.send_data_signal.connect(connection.send_data)

    @pyqtSlot(object)
    def on_data_connected_to_server(self, connection):
        self.connection = connection
        connection.data_received_signal.connect(self.on_data_received)
        self.send_data_signal.connect(connection.send_data)
        self.log_signal.emit(f"Ожидание данных для управления роботом", LogMessageType.WARNING)

    @pyqtSlot(object)
    def on_data_client_disconnected(self, connection):
        self.connection = None
        connection.data_received_signal.disconnect(self.on_data_received)
        self.send_data_signal.disconnect(connection.send_data)

    @pyqtSlot(object)
    def on_data_disconnected_from_server(self, connection):
        self.connection = None
        connection.data_received_signal.disconnect(self.on_data_received)
        self.send_data_signal.disconnect(connection.send_data)

    @pyqtSlot(object)
    def on_data_received(self, data):
        # Получены данные по сети
        #if not self.connection.is_server: # Мы на стороне робота
        #    print(data)
        self.processEvents()

if __name__ == "__main__":
    app = QRobotApplication(sys.argv)
    _main_window = QRobotMainWindow(app)
    _main_window.show()
    app.start(_main_window)
    app.exec()
    app.stop()