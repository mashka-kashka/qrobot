from time import localtime, strftime
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QTextFormat, QColor, QTextCursor, QPixmap, QImage
from PyQt6.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QLabel, QSlider
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot

from main_window_ui import Ui_MainWindow
from log_message_type import LogMessageType
from net_config_dialog import NetConfigDialog
import toml


class QRobotMainWindow(QMainWindow):
    start_server_signal = pyqtSignal()
    stop_server_signal = pyqtSignal()
    reset_server_signal = pyqtSignal()
    start_client_signal = pyqtSignal()
    stop_client_signal = pyqtSignal()
    reset_client_signal = pyqtSignal()

    def __init__(self, app):
        super().__init__()

        self.app = app

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.logger = self.ui.teLog

        self.scene = QGraphicsScene()
        self.ui.gv_camera.setScene(self.scene)
        self.scenePixmapItem = None

        with open('config.toml', 'r') as f:
            self.config = toml.load(f)

            servos = self.config["servos"]
            channels = servos["channels"]
            descriptions = servos["descriptions"]
            positions = servos["positions"]
            mins = servos["mins"]
            maxs = servos["maxs"]

            layout = self.ui.gl_servos
            for row, description in enumerate(descriptions):
                label = QLabel(description)
                layout.addWidget(label, row, 0)
                slider = QSlider(QtCore.Qt.Orientation.Horizontal)
                slider.setSliderPosition(positions[row])
                slider.setMinimum(mins[row])
                slider.setMaximum(maxs[row])
                slider.valueChanged.connect(self.on_slider_value_changed)
                slider.setProperty("id", row)
                slider.setProperty("channel", channels[row])
                layout.addWidget(slider, row, 1)

        app = QtWidgets.QApplication.instance()
        app.show_frame_signal.connect(self.show_frame)

    @pyqtSlot()
    def on_slider_value_changed(self):
        slider = self.sender()
        id = slider.property("id")
        #print(f"Сервопривод: {id} Значение: {slider.value()}")
        pass

    @pyqtSlot(object)
    def show_frame(self, frame):
        if not isinstance(frame, QImage):
            frame = QImage(
                 frame.data,
                 frame.shape[1],
                 frame.shape[0],
                 QImage.Format.Format_BGR888,
            )

        _pixmap = QPixmap.fromImage(frame)

        if self.scenePixmapItem is None:
            self.scenePixmapItem = QGraphicsPixmapItem(_pixmap)
            self.scene.addItem(self.scenePixmapItem)
            self.scenePixmapItem.setZValue(0)
        else:
            self.scenePixmapItem.setPixmap(_pixmap)

        self.ui.gv_camera.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.ui.gv_camera.show()

    def hide_image(self):
        self.ui.gv_camera.hide()

    def log(self, message, type=LogMessageType.STATUS):
        fmt = QTextFormat()
        self.logger.moveCursor(QTextCursor.MoveOperation.End)
        if type == LogMessageType.ERROR:
            self.logger.setTextColor(QColor(255, 0, 0))
        elif type == LogMessageType.WARNING:
            self.logger.setTextColor(QColor(0, 0, 255))
        else:
            self.logger.setTextColor(QColor(0, 0, 0))
        self.logger.append(strftime("%H:%M:%S : ", localtime()))
        self.logger.insertPlainText(message)

    def on_config(self):
        _dialog = NetConfigDialog()
        with open('config.toml', 'r') as f:
            self.config = toml.load(f)
            _dialog.set_video_port(self.config["network"]["video_port"])
            _dialog.set_data_port(self.config["network"]["data_port"])
            _dialog.set_host(self.config["network"]["host"])
        if _dialog.exec():
            self.config["network"]["video_port"] = _dialog.get_video_port()
            self.config["network"]["data_port"] = _dialog.get_data_port()
            self.config["network"]["host"] =_dialog.get_host()
            with open('config.toml', 'w') as f:
                toml.dump(self.config, f)

            if self.ui.actionActivateComputer.isEnabled():
                self.reset_server_signal.emit()
            else:
                self.reset_client_signal.emit()

    @pyqtSlot(bool)
    def activate_robot(self, block_signals):
        self.ui.actionActivateRobot.blockSignals(block_signals)
        self.ui.actionActivateRobot.toggle()
        self.ui.actionActivateRobot.blockSignals(not block_signals)

    @pyqtSlot(bool)
    def on_activate_robot(self, start):
        _translate = QtCore.QCoreApplication.translate
        if start:
            self.ui.actionActivateRobot.setStatusTip(_translate("MainWindow", "Отключить робота"))
            self.ui.actionActivateRobot.setText(_translate("MainWindow", "Отключить робота"))
            self.ui.actionActivateRobot.setIconText(_translate("MainWindow", "Отключить робота"))
            self.ui.actionActivateComputer.setEnabled(False)
            self.start_client_signal.emit()
        else:
            self.ui.actionActivateRobot.setStatusTip(_translate("MainWindow", "Активировать робота"))
            self.ui.actionActivateRobot.setText(_translate("MainWindow", "Активировать робота"))
            self.ui.actionActivateRobot.setIconText(_translate("MainWindow", "Активация робота"))
            self.ui.actionActivateComputer.setEnabled(True)
            self.stop_client_signal.emit()

    @pyqtSlot(bool)
    def activate_computer(self, block_signals):
        self.ui.actionActivateComputer.blockSignals(block_signals)
        self.ui.actionActivateComputer.toggle()
        self.ui.actionActivateComputer.blockSignals(not block_signals)

    @pyqtSlot(bool)
    def on_activate_computer(self, start):
        _translate = QtCore.QCoreApplication.translate
        if start:
            self.ui.actionActivateComputer.setStatusTip(_translate("MainWindow", "Отключить компьютер"))
            self.ui.actionActivateComputer.setText(_translate("MainWindow", "Отключить компьютер"))
            self.ui.actionActivateComputer.setIconText(_translate("MainWindow", "Отключить компьютер"))
            self.ui.actionActivateRobot.setEnabled(False)
            self.start_server_signal.emit()
        else:
            self.ui.actionActivateComputer.setStatusTip(_translate("MainWindow", "Активировать компьютер"))
            self.ui.actionActivateComputer.setText(_translate("MainWindow", "Активировать компьютер"))
            self.ui.actionActivateComputer.setIconText(_translate("MainWindow", "Активация компьютера"))
            self.ui.actionActivateRobot.setEnabled(True)
            self.app.camera.start()
            self.stop_server_signal.emit()
