from time import localtime, strftime
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QTextFormat, QColor, QTextCursor, QPixmap, QImage
from PyQt6.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QLabel, QSlider, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot
from torchgen.api.types import layoutT

from servo_controller import QServoController
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

        layout = self.ui.gl_servos
        btn_reset = QPushButton("Сброс")
        btn_reset.clicked.connect(self.on_reset_servos)
        layout.addWidget(btn_reset, 0, 0)

        self.sliders = []
        self.controller = app.controller
        for id in range(self.controller.get_servos_count()):
            channel, name, begin, end, neutral = self.controller.get_servo_info(id)
            label = QLabel(name)
            layout.addWidget(label, id + 1, 0)
            slider = QSlider(QtCore.Qt.Orientation.Horizontal)
            reverse = begin > end
            slider.setProperty("reverse", reverse)
            slider.setMinimum(end if reverse else begin)
            slider.setMaximum(begin if reverse else end)
            slider.setValue(neutral)
            slider.valueChanged.connect(self.on_slider_value_changed)
            slider.setProperty("id", id + 1)
            slider.setProperty("channel", channel)
            slider.setProperty("begin", begin)
            slider.setProperty("end", end)
            slider.setProperty("neutral", neutral)
            layout.addWidget(slider, id + 1, 1)
            self.sliders.append(slider)

        app = QtWidgets.QApplication.instance()
        app.show_frame_signal.connect(self.show_frame)

    @pyqtSlot()
    def on_reset_servos(self):
        for slider in self.sliders:
            neutral_val = slider.property("neutral")
            slider.setValue(neutral_val)

    @pyqtSlot()
    def on_slider_value_changed(self):
        slider = self.sender()
        channel = slider.property("channel")
        begin_val = slider.property("begin")
        end_val = slider.property("end")
        reverse = slider.property("reverse")
        value = slider.value()
        if reverse:
            value = begin_val - value + end_val

        self.controller.set_servo_position(channel, value, True)

        #print(f"Сервопривод: {channel} Значение: {value}")

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
