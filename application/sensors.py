from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import numpy as np


class QRobotSensors(QObject):
    data_captured_signal = pyqtSignal(object, object, object, object)
    running = False

    def __init__(self):
        super().__init__()

    @pyqtSlot()
    def start(self):
        self.running = True
        self.get_data() # Получение первого кадра

    def stop(self):
        self.running = False

    @pyqtSlot()
    def get_data(self):
        if not self.running:
            return

        data = np.linspace(start=20, stop=40, num=768)
        data.shape = (32, 24)
        try:
            self.data_captured_signal.emit(data, 30, 40, 50)
        except:
            pass


