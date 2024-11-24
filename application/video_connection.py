import sys
from socket import socket

from PyQt6 import QtWidgets
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, QByteArray, QBuffer
from PyQt6.QtNetwork import QTcpServer, QHostAddress, QTcpSocket
from log_message_type import LogMessageType
import pickle

class QRobotVideoConnection(QThread):
    log_signal = pyqtSignal(object, object)
    stop_signal = pyqtSignal()
    frame_received_signal = pyqtSignal(object)
    connected_signal = pyqtSignal(object)
    disconnected_signal = pyqtSignal(object)
    tcp_server = None
    tcp_client = None
    connection = None

    def __init__(self, logger, host, port):
        super().__init__()
        self.logger = logger
        self.log_signal.connect(self.logger.log)
        self.host = host
        self.port = int(port)

        app = QtWidgets.QApplication.instance()
        self.frame_received_signal.connect(app.on_frame_received)
        self.connected_signal.connect(app.on_connected)
        self.disconnected_signal.connect(app.on_disconnected)

    @pyqtSlot()
    def bind(self):
        try:
            self.tcp_server = QTcpServer()
            self.tcp_server.newConnection.connect(self.on_client_connected)
            self.log_signal.emit(f"Ожидание подключения клиента к {self.host}:{self.port} для передачи видео",
                                 LogMessageType.STATUS)

            self.tcp_server.listen(QHostAddress(self.host), self.port)
        except Exception as e:
            self.log_signal.emit(f"Ошибка {type(e)}: {e}", LogMessageType.ERROR)
            self.stop_signal.emit()

    @pyqtSlot()
    def connect_to_host(self):
        try:
            self.log_signal.emit(f"Попытка подключения к серверу {self.host}:{self.port} для передачи видео",
                                 LogMessageType.STATUS)
            self.tcp_client = QTcpSocket()
            self.tcp_client.disconnected.connect(self.on_client_disconnected)
            self.tcp_client.connected.connect(self.on_connected_to_host)
            self.tcp_client.connectToHost(QHostAddress(self.host), self.port)
            if not self.tcp_client.waitForConnected(1000):
                self.log_signal.emit(f"Не удалось подключиться к серверу для передачи видео", LogMessageType.WARNING)
                self.stop_signal.emit()
        except Exception as e:
            self.log_signal.emit(f"Ошибка {type(e)}: {e}", LogMessageType.ERROR)
            self.stop_signal.emit()

    def close(self):
        try:
            if self.tcp_server:
                self.log_signal.emit(f"Остановка сервера {self.host}:{self.port} для передачи видео",
                                     LogMessageType.STATUS)
                self.tcp_server.close()
                self.tcp_server = None

            if self.tcp_client:
                self.log_signal.emit(f"Остановка клиента {self.host}:{self.port} для передачи видео",
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
            self.connection.disconnected.connect(self.on_client_disconnected)
            self.log_signal.emit(f"Подключился клиент {self.connection.peerAddress().toString()} для передачи видео",
                             LogMessageType.STATUS)
            self.connected_signal.emit(self)

    @pyqtSlot()
    def on_connected_to_host(self):
        self.log_signal.emit(f"Успешное подключение к серверу для передачи видео",
                             LogMessageType.STATUS)
        self.connected_signal.emit(self)
        self.tcp_client.readyRead.connect(self.receive_frame)
        self.bytes_expected = 0

    @pyqtSlot()
    def on_client_disconnected(self):
        self.log_signal.emit(f"Клиент для передачи видео отключился",
                         LogMessageType.WARNING)
        self.disconnected_signal.emit(self)

    @pyqtSlot(object)
    def send_frame(self, frame):
        try:
            if self.connection.state() == QTcpSocket.SocketState.ConnectedState:
                _buffer = pickle.dumps(frame)
                _size = QByteArray()
                _size.setNum(len(_buffer))
                self.connection.write(_size + b'\n')
                self.connection.write(_buffer)
                self.connection.waitForBytesWritten()
        except Exception as e:
            self.log_signal.emit(f"send_frame >> Ошибка {type(e)}: {e}", LogMessageType.ERROR)

    @pyqtSlot()
    def receive_frame(self):
        try:
            while self.tcp_client.bytesAvailable() > 0:
                if self.bytes_expected == 0 and self.tcp_client.bytesAvailable() >= self.bytes_expected:
                    chunk = self.tcp_client.readLine()
                    self.bytes_expected = int(chunk)
                    self.buffer = QByteArray()

                if self.bytes_expected > 0 and self.tcp_client.bytesAvailable() > 0:
                    chunk = self.tcp_client.read(min(self.bytes_expected, self.tcp_client.bytesAvailable()))
                    self.bytes_expected -= len(chunk)
                    self.buffer.append(chunk)
                    if self.bytes_expected == 0:
                        _frame = pickle.loads(self.buffer)
                        self.frame_received_signal.emit(_frame)
        except Exception as e:
            self.log_signal.emit(f"Ошибка {type(e)}: {e}", LogMessageType.ERROR)

