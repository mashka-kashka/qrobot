from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QThread, QByteArray
from PyQt6.QtNetwork import QTcpServer, QHostAddress, QTcpSocket
from log_message_type import LogMessageType
import json


class QRobotDataConnection(QThread):
    log_signal = pyqtSignal(object, object)
    data_stop_signal = pyqtSignal()
    data_received_signal = pyqtSignal(object)
    data_client_connected_signal = pyqtSignal(object)
    data_client_disconnected_signal = pyqtSignal(object)
    data_connected_to_server_signal = pyqtSignal(object)
    data_disconnected_from_server_signal = pyqtSignal(object)
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
        self.data_received_signal.connect(app.on_data_received)
        self.data_client_connected_signal.connect(app.on_data_client_connected)
        self.data_client_disconnected_signal.connect(app.on_data_client_disconnected)
        self.data_connected_to_server_signal.connect(app.on_data_connected_to_server)
        self.data_disconnected_from_server_signal.connect(app.on_data_disconnected_from_server)

    @pyqtSlot()
    def bind(self):
        try:
            self.tcp_server = QTcpServer()
            self.tcp_server.newConnection.connect(self.on_client_connected)
            self.log_signal.emit(f"Ожидание подключения клиента к {self.host}: {self.port} для передачи данных",
                                 LogMessageType.STATUS)
            self.tcp_server.listen(QHostAddress(self.host), self.port)
        except Exception as e:
            self.log_signal.emit(f"Ошибка {type(e)}: {e}", LogMessageType.ERROR)
            self.data_stop_signal.emit()

    @pyqtSlot()
    def connect_to_server(self):
        try:
            self.log_signal.emit(f"Попытка подключения к серверу {self.host}: {self.port} для передачи данных",
                                 LogMessageType.STATUS)
            self.tcp_client = QTcpSocket()
            self.tcp_client.disconnected.connect(self.on_disconnected_from_server)
            self.tcp_client.connected.connect(self.on_connected_to_server)
            self.tcp_client.connectToHost(QHostAddress(self.host), self.port)
            if not self.tcp_client.waitForConnected(2000):
                self.log_signal.emit(f"Не удалось подключиться к серверу для передачи данных", LogMessageType.WARNING)
                self.data_stop_signal.emit()
        except Exception as e:
            self.log_signal.emit(f"Ошибка {type(e)}: {e}", LogMessageType.ERROR)
            self.data_stop_signal.emit()

    def close(self):
        try:
            if self.tcp_server:
                self.log_signal.emit(f"Остановка сервера {self.host}: {self.port} для передачи данных",
                                     LogMessageType.STATUS)
                self.tcp_server.close()
                self.tcp_server = None

            if self.tcp_client:
                self.log_signal.emit(f"Остановка клиента {self.host}: {self.port} для передачи данных",
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
            self.connection.readyRead.connect(self.receive_data)
            self.connection.disconnected.connect(self.on_client_disconnected)
            self.log_signal.emit(f"Подключился клиент {self.connection.peerAddress().toString()} для передачи данных",
                             LogMessageType.STATUS)
            self.data_client_connected_signal.emit(self)

    @pyqtSlot()
    def on_client_disconnected(self):
        self.log_signal.emit(f"Клиент для передачи данных отключился",
                         LogMessageType.WARNING)
        self.data_client_disconnected_signal.emit(self)

    @pyqtSlot()
    def on_connected_to_server(self):
        self.log_signal.emit(f"Успешное подключение к серверу для передачи данных",
                             LogMessageType.STATUS)
        self.tcp_client.readyRead.connect(self.receive_data)
        self.bytes_expected = 0
        self.data_connected_to_server_signal.emit(self)

    @pyqtSlot()
    def on_disconnected_from_server(self):
        self.log_signal.emit(f"Произошло отключение от сервера для передачи данных",
                             LogMessageType.STATUS)
        self.tcp_client.close()
        self.data_disconnected_from_server_signal.emit(self)

    @pyqtSlot(object)
    def send_data(self, data):
        try:
            connection = self.connection if self.connection else self.tcp_client

            _buffer = json.dumps(data).encode('utf-8')
            _size = QByteArray()
            _size.setNum(len(_buffer))
            connection.write(_size + b'\n')
            connection.write(_buffer)
            connection.waitForBytesWritten(10)
            connection.flush()
        except Exception as e:
            self.log_signal.emit(f"Ошибка передачи данных {type(e)}: {e}", LogMessageType.ERROR)

    @pyqtSlot()
    def receive_data(self):
        try:
            connection = self.connection if self.connection else self.tcp_client
            while connection.bytesAvailable() > 0:
                if self.bytes_expected == 0 and connection.bytesAvailable() >= self.bytes_expected:
                    chunk = connection.readLine()
                    self.bytes_expected = int(chunk)
                    self.buffer = QByteArray()

                if self.bytes_expected > 0 and connection.bytesAvailable() > 0:
                    chunk = connection.read(min(self.bytes_expected, connection.bytesAvailable()))
                    self.bytes_expected -= len(chunk)
                    self.buffer.append(chunk)
                    if self.bytes_expected == 0:
                        connection.flush()
                        _data = json.loads(str(self.buffer.data(), encoding = 'utf-8'))
                        self.data_received_signal.emit(_data)
        except Exception as e:
            self.log_signal.emit(f"Ошибка получения данных {type(e)}: {e}", LogMessageType.ERROR)

