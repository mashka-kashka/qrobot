from time import localtime, strftime
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QTextFormat, QColor, QTextCursor, QPixmap, QImage, qRgb
from PyQt6.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QLabel, QSlider, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot
from main_window_ui import Ui_MainWindow
from log_message_type import LogMessageType
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

class QRobotMainWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()

        self.app = app

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.logger = self.ui.teLog

        # Камера
        self.camera_scene = QGraphicsScene()
        self.ui.gv_camera.setScene(self.camera_scene)
        self.cameraScenePixmapItem = None

        # Тепловизор
        self.thermo_scene = QGraphicsScene()
        self.ui.gv_thermo.setScene(self.thermo_scene)
        self.thermoScenePixmapItem = None
        self.thermoImage = QImage(32, 24, QImage.Format.Format_RGB32)
        self.Tmax = 40
        self.Tmin = 20
        self.norm = mpl.colors.Normalize(vmin=self.Tmin, vmax=self.Tmax, clip=True)
        self.cmap = cm.get_cmap('viridis')
        self.mapper = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

        servos_layout = self.ui.gl_servos
        btn_reset_servos = QPushButton("Сброс")
        btn_reset_servos.clicked.connect(self.on_reset_servos)
        servos_layout.addWidget(btn_reset_servos, 0, 0)

        self.sliders = []
        self.servo_controller = app.robot.controller
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

    @pyqtSlot(object, object, object, object)
    def show_sensors_data(self, thermo, tfinger, spo2, pulse):
        for y in range(24):
            for x in range(32):
                value = thermo[x, y]
                r, g, b, a = self.mapper.to_rgba(value, bytes=True)
                self.thermoImage.setPixel(x, y, qRgb(r, g, b))

        _pixmap = QPixmap.fromImage(self.thermoImage)

        if self.thermoScenePixmapItem is None:
            self.thermoScenePixmapItem = QGraphicsPixmapItem(_pixmap)
            self.thermo_scene.addItem(self.thermoScenePixmapItem)
            self.thermoScenePixmapItem.setZValue(0)
        else:
            self.thermoScenePixmapItem.setPixmap(_pixmap)

        self.ui.gv_thermo.fitInView(self.thermo_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.ui.gv_thermo.show()

        self.ui.lbMaxTemperature.setText(f"Температура в кадре от {thermo.min()} до {thermo.max()}")
        self.ui.lbFinger.setText(f"Температура пальца: {tfinger}")
        self.ui.lbPulse.setText(f"Пульс: {pulse}")
        self.ui.lbSPO2.setText(f"Сатурация: {spo2}")

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

        if self.cameraScenePixmapItem is None:
            self.cameraScenePixmapItem = QGraphicsPixmapItem(_pixmap)
            self.camera_scene.addItem(self.cameraScenePixmapItem)
            self.cameraScenePixmapItem.setZValue(0)
        else:
            self.cameraScenePixmapItem.setPixmap(_pixmap)

        self.ui.gv_camera.fitInView(self.camera_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
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