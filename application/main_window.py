from time import localtime, strftime
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QTextFormat, QColor, QTextCursor, QPixmap, QImage
from PyQt6.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QLabel, QSlider, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot
from main_window_ui import Ui_MainWindow
from log_message_type import LogMessageType


class QRobotMainWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()

        self.app = app

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.logger = self.ui.teLog

        self.scene = QGraphicsScene()
        self.ui.gv_camera.setScene(self.scene)
        self.scenePixmapItem = None

        servos_layout = self.ui.gl_servos
        btn_reset_servos = QPushButton("Сброс")
        btn_reset_servos.clicked.connect(self.on_reset_servos)
        servos_layout.addWidget(btn_reset_servos, 0, 0)

        self.sliders = []
        self.servo_controller = app.servo_controller
        for id in range(self.servo_controller.get_servos_count()):
            channel, name, begin, end, neutral = self.servo_controller.get_servo_info(id)
            label = QLabel(name)
            servos_layout.addWidget(label, id + 1, 0)
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
            servos_layout.addWidget(slider, id + 1, 1)
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

        self.servo_controller.set_servo_position(channel, value, True)

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