from venv import logger

from PyQt6.QtCore import QObject, pyqtSignal, QThread, pyqtSlot
from video_connection import QRobotVideoConnection
from log_message_type import LogMessageType
import toml


class QRobotServer(QObject):
    log_signal = pyqtSignal(object, object)
    stop_signal = pyqtSignal()
    running = False

    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.log_signal.connect(self.logger.log)

    def start(self):
        if self.running:
            return
        self.running = True

        with open('config.toml', 'r') as f:
            self.config = toml.load(f)
            _video_port = int(self.config["network"]["video_port"])
            _data_port = int(self.config["network"]["data_port"])
            _host = self.config["network"]["host"]

            self.video_connection = QRobotVideoConnection(self.logger, _host, _video_port)
            self.video_connection.stop_signal.connect(self.on_stop)
            self.video_connection.started.connect(self.video_connection.bind)
            self.video_connection.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        try:
            self.video_connection.close()
        except Exception as e:
            self.log_signal.emit(f"Ошибка {type(e)}: {e}", LogMessageType.ERROR)

    def reset(self):
        self.stop()
        self.start()

    @pyqtSlot()
    def on_stop(self):
        self.stop_signal.emit()
