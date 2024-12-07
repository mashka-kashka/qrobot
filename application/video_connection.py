from PyQt6 import QtWidgets
from PyQt6.QtGui import QImage
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QThread, QByteArray
from PyQt6.QtNetwork import QTcpServer, QHostAddress, QTcpSocket
from log_message_type import LogMessageType
import numpy as np
import pickle
import time

class QRobotVideoConnection(QThread):
    log_signal = pyqtSignal(object, object)
    video_stop_signal = pyqtSignal()
    frame_received_signal = pyqtSignal(object)
    video_client_connected_signal = pyqtSignal(object)
    video_client_disconnected_signal = pyqtSignal(object)
    video_connected_to_server_signal = pyqtSignal(object)
    video_disconnected_from_server_signal = pyqtSignal(object)
    tcp_server = None
    tcp_client = None
    connection = None

    def __init__(self, logger, host, port, is_server):
        super().__init__()
        self.logger = logger
        self.log_signal.connect(self.logger.log)
        self.host = host
        self.port = int(port)
        self.is_server = is_server
        self.bytes_expected = 0

        app = QtWidgets.QApplication.instance()
        self.frame_received_signal.connect(app.on_frame_received)
        self.video_client_connected_signal.connect(app.on_video_client_connected)
        self.video_client_disconnected_signal.connect(app.on_video_client_disconnected)
        self.video_connected_to_server_signal.connect(app.on_video_connected_to_server)
        self.video_disconnected_from_server_signal.connect(app.on_video_disconnected_from_server)

    @pyqtSlot()
    def bind(self):
        try:
            self.tcp_server = QTcpServer()
            self.tcp_server.newConnection.connect(self.on_client_connected)
            self.log_signal.emit(f"Ожидание подключения клиента к {self.host}: {self.port} для передачи видео",
                                 LogMessageType.STATUS)
            self.tcp_server.listen(QHostAddress(self.host), self.port)
        except Exception as e:
            self.log_signal.emit(f"Ошибка {type(e)}: {e}", LogMessageType.ERROR)
            self.video_stop_signal.emit()

    @pyqtSlot()
    def connect_to_server(self):
        try:
            self.log_signal.emit(f"Попытка подключения к серверу {self.host}: {self.port} для передачи видео",
                                 LogMessageType.STATUS)
            self.tcp_client = QTcpSocket()
            self.tcp_client.disconnected.connect(self.on_disconnected_from_server)
            self.tcp_client.connected.connect(self.on_connected_to_server)
            self.tcp_client.connectToHost(QHostAddress(self.host), self.port)
            if not self.tcp_client.waitForConnected(2000):
                self.log_signal.emit(f"Не удалось подключиться к серверу для передачи видео", LogMessageType.WARNING)
                self.video_stop_signal.emit()
        except Exception as e:
            self.log_signal.emit(f"Ошибка {type(e)}: {e}", LogMessageType.ERROR)
            self.video_stop_signal.emit()

    def close(self):
        try:
            if self.tcp_server:
                self.log_signal.emit(f"Остановка сервера {self.host}: {self.port} для передачи видео",
                                     LogMessageType.STATUS)
                self.tcp_server.close()
                self.tcp_server = None

            if self.tcp_client:
                self.log_signal.emit(f"Остановка клиента {self.host}: {self.port} для передачи видео",
                                     LogMessageType.STATUS)
                self.tcp_client.close()
                self.tcp_client = None

            self.quit()

        except Exception as e:
           self.log_signal.emit(f"Ошибка {type(e)}: {e}", LogMessageType.ERROR)

    @pyqtSlot()
    def on_client_connected(self):
        while self.tcp_server.hasPendingConnections():
            self.connection = self.tcp_server.nextPendingConnection()
            self.bytes_expected = 0
            self.connection.readyRead.connect(self.receive_frame)
            self.connection.disconnected.connect(self.on_client_disconnected)
            self.log_signal.emit(f"Подключился клиент {self.connection.peerAddress().toString()} для передачи видео",
                             LogMessageType.STATUS)
            self.video_client_connected_signal.emit(self)

    @pyqtSlot()
    def on_client_disconnected(self):
        self.log_signal.emit(f"Клиент для передачи видео отключился",
                         LogMessageType.WARNING)
        self.video_client_disconnected_signal.emit(self)

    @pyqtSlot()
    def on_connected_to_server(self):
        self.log_signal.emit(f"Успешное подключение к серверу для передачи видео",
                             LogMessageType.STATUS)
        self.tcp_client.readyRead.connect(self.receive_frame)
        self.bytes_expected = 0
        self.video_connected_to_server_signal.emit(self)

    @pyqtSlot()
    def on_disconnected_from_server(self):
        self.log_signal.emit(f"Произошло отключение от сервера для передачи видео",
                             LogMessageType.STATUS)
        self.tcp_client.close()
        self.video_disconnected_from_server_signal.emit(self)

    def QImageToCvMat(self, incomingImage):
        width = incomingImage.width()
        height = incomingImage.height()

        ptr = incomingImage.bits()
        ptr.setsize(height * width * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        return arr

    @pyqtSlot(object)
    def send_frame(self, frame):
        try:
            connection = self.tcp_client
            if self.connection:
                connection = self.connection

            if isinstance(frame, QImage):
                frame = self.QImageToCvMat(frame)

            _buffer = pickle.dumps(frame)
            _time = QByteArray()
            _time.setNum(time.time_ns())
            connection.write(_time + b'\n')
            _size = QByteArray()
            _size.setNum(len(_buffer))
            connection.write(_size + b'\n')
            connection.write(_buffer)
            connection.waitForBytesWritten(100)
            connection.flush()
        except Exception as e:
            self.log_signal.emit(f"Ошибка передачи кадра {type(e)}: {e}", LogMessageType.ERROR)

    @pyqtSlot()
    def receive_frame(self):
        try:
            connection = self.tcp_client
            if self.connection:
                connection = self.connection
            while connection.bytesAvailable() > 0:
                if self.bytes_expected == 0 and connection.bytesAvailable() >= self.bytes_expected:
                    chunk = connection.readLine()
                    self.send_time = int(chunk)
                    chunk = connection.readLine()
                    self.bytes_expected = int(chunk)
                    self.buffer = QByteArray()

                if self.bytes_expected > 0 and connection.bytesAvailable() > 0:
                    chunk = connection.read(min(self.bytes_expected, connection.bytesAvailable()))
                    self.bytes_expected -= len(chunk)
                    self.buffer.append(chunk)
                    if self.bytes_expected == 0:
                        connection.flush()
                        _frame = pickle.loads(self.buffer)
                        cur_time = time.time_ns()
                        #print(f"send:{self.send_time} cur:{cur_time} delta: {cur_time - self.send_time}")
                        delta = cur_time - self.send_time
                        if delta < 50000000:
                            self.frame_received_signal.emit(_frame)
        except Exception as e:
            self.log_signal.emit(f"Ошибка получения кадра {type(e)}: {e}", LogMessageType.ERROR)

