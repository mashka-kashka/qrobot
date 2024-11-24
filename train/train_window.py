from operator import lshift
from random import sample
from random import sample
from time import time, localtime, strftime
from PyQt6.QtGui import QTextFormat, QColor, QTextCursor, QPixmap, QIcon, QPainter, QFont, QImage
from PyQt6.QtWidgets import (QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFileDialog, QMessageBox,
                             QVBoxLayout)
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot, QModelIndex, QPoint
from tensorflow.python.ops.gen_batch_ops import batch
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.metrics_impl import accuracy
from torch import layout

from models.emotions_model import EmotionsNet
from train.torch_dataset import TorchDataset
from models.gestures_model import GesturesNet
from train.labels_data_table_model import LabelsDataTableModel
from train.train_data_table_model import TrainDataTableModel
from train_window_ui import Ui_TrainWindow
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import pyqtgraph as pg

class TrainWindow(QMainWindow):
    TRAIN_TAB=2

    DEFAULT_MODE = 0
    GESTURES_MODE = 1
    EMOTIONS_MODE = 2

    label_font = QFont("Times", 20)
    emoji_font = QFont("Noto Color Emoji", 64)
    model_filename = ''

    def __init__(self, app):
        super().__init__()

        self.app = app
        self.mode = self.DEFAULT_MODE

        # Интерфейс
        self.ui = Ui_TrainWindow()
        self.ui.setupUi(self)

        self.pixmap = None
        self.input_scene = QGraphicsScene()
        self.ui.gv_input.setScene(self.input_scene)
        self.classification_scene = QGraphicsScene()
        self.ui.gv_classification.setScene(self.classification_scene)

        layout = QVBoxLayout()
        self.train_progress_graph = pg.PlotWidget()
        self.train_progress_graph.setBackground("w")
        self.train_progress_graph.setLabel(
            "bottom",
            '<span style="color: red; font-size: 14px">Эпохи</span>'
        )
        self.train_progress_graph.addLegend()
        self.train_loss_plot = pg.PlotCurveItem(clear=True, pen="r", name="Ошибка на обучающей выборке")
        self.valid_loss_plot = pg.PlotCurveItem(clear=True, pen="g", name="Ошибка на тестовой выборке")
        self.accuracy_plot = pg.PlotCurveItem(clear=True, pen="b", name="Точность")
        self.train_progress_graph.addItem(self.train_loss_plot)
        self.train_progress_graph.addItem(self.valid_loss_plot)
        self.train_progress_graph.addItem(self.accuracy_plot)
        layout.addWidget(self.train_progress_graph)
        self.ui.fr_progress.setLayout(layout)

        self.logger = self.ui.teLog

        # Модель таблицы данных для меток
        self.labels_data_table_model = LabelsDataTableModel()

        # Модель таблицы данных для обучения
        self.train_data_model = TrainDataTableModel()
        self.ui.tv_data.setModel(self.train_data_model)
        self.train_data_model.dataChanged.connect(self.on_data_changed)
        self.train_data_model.modelReset.connect(self.on_data_changed)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Нейронная сеть
        self.model = None

    def get_mode(self):
        return self.mode

    def is_palm_visible(self):
        return not self.ui.gv_palm.visibleRegion().isEmpty()

    @pyqtSlot(object)
    def show_frame(self, frame, results):
        image = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            QImage.Format.Format_BGR888,
        )

        # Активна вкладка "Распознавание"
        if self.model and self.ui.tabClassification.isVisible():
            try:
                sample = self.train_data_model.get_sample(results, True)
                if sample:
                    input = torch.unsqueeze(torch.tensor(sample).double(), dim=0).to(self.device)
                    prediction = self.model(input)
                    score = max(prediction[0])
                    label = self.model.get_label(prediction)
                    label = self.labels_data_table_model.get_unicode_by_id(label)

                    painter = QPainter(image)
                    painter.setPen(QColor(255, 255, 255))
                    painter.setFont(self.label_font)
                    painter.drawText(QPoint(5, 25), f"Уверенность: {score:.4f}")
                    #if score > 0.7:
                    painter.setFont(self.emoji_font)
                    painter.drawText(QPoint(5, 100), f"{label}")
                    painter.end()
            except Exception as e:
                print(e)

        _pixmap = QPixmap.fromImage(image)

        if self.pixmap is None:
            self.pixmap = QGraphicsPixmapItem(_pixmap)
            self.pixmap.setZValue(0)
        else:
            self.pixmap.setPixmap(_pixmap)

        # Активна вкладка "Распознавание"
        if self.ui.tabClassification.isVisible():
            if len(self.classification_scene.items()) == 0:
                self.classification_scene.addItem(self.pixmap)
            self.ui.gv_classification.fitInView(self.classification_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.ui.gv_classification.show()

        # Активна вкладка "Данные"
        elif self.ui.tabData.isVisible():
            if len(self.input_scene.items()) == 0:
                self.input_scene.addItem(self.pixmap)
            self.ui.gv_input.fitInView(self.input_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.ui.gv_input.show()

    def log(self, message, color):
        fmt = QTextFormat()
        self.logger.moveCursor(QTextCursor.MoveOperation.End)
        self.logger.setTextColor(color)
        self.logger.append(strftime("%H:%M:%S : ", localtime()))
        self.logger.insertPlainText(message)

    @pyqtSlot()
    def on_add_sample(self):
        sample_index = self.ui.cb_labels.currentData(Qt.ItemDataRole.UserRole)
        if self.mode == TrainWindow.GESTURES_MODE:
            self.train_data_model.add(sample_index, self.app.hand_results)
        else:
            self.train_data_model.add(sample_index, self.app.face_results)

    @pyqtSlot()
    def on_new_data(self):
        if self.train_data_model.is_modified():
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Сохранить изменённые данные перед созданием новых?")
            dlg.setText("Данные изменены!")
            dlg.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            dlg.setIcon(QMessageBox.Icon.Question)
            button = dlg.exec()

            if button == QMessageBox.StandardButton.Yes:
                self.log("TODO: Сохранение изменённых данных", QColor(255,0,0))

        self.train_data_model.create(self.labels_data_table_model.get_type())

    @pyqtSlot()
    def on_open_data(self):
        fname = QFileDialog.getOpenFileName(
            self,
            "Открыть файл данных для обучения модели",
            ".",
            "Текст CSV (*.csv);; Все файлы (*.*)",
        )
        if fname[0]:
            self.train_data_model.open(fname[0])

    @pyqtSlot()
    def on_save_data(self):
        filename = self.train_data_model.get_file_name()
        if filename:
            self.train_data_model.save(filename)
        else:
            self.on_save_data_as()

    @pyqtSlot()
    def on_save_data_as(self):
        fname = QFileDialog.getSaveFileName(
            self,
            "Открыть файл данных для обучения модели",
            ".",
            "Текст CSV (*.csv);; Все файлы (*.*)",
        )
        if fname[0]:
            self.train_data_model.save(fname[0])

    @pyqtSlot()
    def on_data_changed(self):
        self.ui.tb_save_data.setEnabled(self.train_data_model.is_modified())
        self.ui.tb_save_data_as.setEnabled(self.train_data_model.is_modified())

    @pyqtSlot()
    def on_edit_sample(self):
        sample_index = self.ui.tv_data.currentIndex()
        if not sample_index.isValid():
            return
        dlg = QMessageBox(self)
        if self.mode == TrainWindow.EMOTIONS_MODE:
            dlg.setWindowTitle("Изменение эмоции")
        else:
            dlg.setWindowTitle("Изменение жеста")
        idx = sample_index.siblingAtColumn(TrainDataTableModel.ID_COLUMN)
        index = int(self.train_data_model.data(idx, Qt.ItemDataRole.DisplayRole))
        dlg.setStyleSheet('font: "Noto Color Emoji";')
        if self.mode == TrainWindow.EMOTIONS_MODE:
            if self.ui.cb_labels.currentIndex() > 0:
                dlg.setText(f"Заменить эмоцию {self.labels_data_table_model.get_unicode(index) or ''} на {self.ui.cb_labels.currentText()}?")
            else:
                dlg.setText(f"Заменить эмоцию {self.labels_data_table_model.get_unicode(index) or ''} на неопределённую?")
        else:
            if self.ui.cb_labels.currentIndex() > 0:
                dlg.setText(f"Заменить жест {self.labels_data_table_model.get_unicode(index) or ''} на {self.ui.cb_labels.currentText()}?")
            else:
                dlg.setText(f"Заменить жест {self.labels_data_table_model.get_unicode(index) or ''} на неопределённый?")
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if self.mode == TrainWindow.EMOTIONS_MODE:
            dlg.setIconPixmap(self.train_data_model.draw_face(sample_index.row(), 100))
        else:
            dlg.setIconPixmap(self.train_data_model.draw_palm(sample_index.row(), 100))
        button = dlg.exec()

        if button == QMessageBox.StandardButton.Yes:
            self.train_data_model.setData(idx, self.ui.cb_labels.currentIndex(), Qt.ItemDataRole.EditRole)
            self.train_data_model.dataChanged.emit(idx,idx)

    @pyqtSlot()
    def on_delete_sample(self):
        sample_index = self.ui.tv_data.currentIndex()
        if not sample_index.isValid():
            return
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Удаление жеста")
        idx = sample_index.siblingAtColumn(self.train_data_model.GESTURE)
        gesture_index = int(self.train_data_model.data(idx, Qt.ItemDataRole.DisplayRole))
        if self.ui.cb_labels.currentIndex() > 0:
            dlg.setText(
                f"Вы действительно хотите удалить жест {self.labels_data_table_model.get_unicode(gesture_index) or ''}?")
        else:
            dlg.setText(f"Вы действительно хотите удалить неопределённый жест?")
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        dlg.setStyleSheet('font: "Noto Color Emoji";')
        dlg.setIconPixmap(self.train_data_model.draw_palm(sample_index.row(), 100))
        button = dlg.exec()

        if button == QMessageBox.StandardButton.Yes:
            self.train_data_model.removeRow(sample_index.row())

    @pyqtSlot(QModelIndex)
    def on_double_clicked_sample(self, index):
        self.on_edit_sample()

    @pyqtSlot(int)
    def on_tab_changed(self, tab):
        if tab == self.TRAIN_TAB:
            self.ui.tb_train_model.setEnabled(self.train_data_model.rowCount(None) > 0)

    @pyqtSlot()
    def on_train_model(self):
        try:
            y_range = 1.0

            labels = self.train_data_model.get_labels()

            if not self.model or not all(x == y for x, y in zip(labels, self.model.labels)):
                if self.mode == TrainWindow.EMOTIONS_MODE:
                    self.model = EmotionsNet(labels).to(self.device)
                else:
                    self.model = GesturesNet(labels).to(self.device)

            # Разделение полного набора данных на обучающий и тестовых наборы
            test_size = float(self.ui.le_test_size.text())
            X = self.train_data_model.X() # Данные
            y = self.train_data_model.y() # Метки
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=True)

            # Наборы обучающих и тестовых данных
            train_data = TorchDataset(X_train, y_train)
            test_data = TorchDataset(X_test, y_test)

            # Загрузчики наборов данных
            batch_size = int(self.ui.le_batch_size.text())
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

            # Оптимизатор
            learning_rate = float(self.ui.le_learning_rate.text())
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

            # Обучение
            loss_fn = torch.nn.CrossEntropyLoss()
            epochs_count = int(self.ui.le_epochs.text())
            training_losses = []
            valid_losses = []
            accuracies = []
            epochs = []
            for epoch in range(1, epochs_count + 1):
                epochs.append(epoch)
                training_loss = 0.0
                valid_loss = 0.0
                self.model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    output = self.model(inputs)
                    loss = loss_fn(output, targets)
                    loss.backward()
                    optimizer.step()
                    training_loss += loss.data.item() * inputs.size(0)
                training_loss /= len(train_loader.dataset)
                training_losses.append(training_loss)

                if training_loss > y_range:
                    y_range = training_loss

                self.model.eval()
                num_correct = 0
                num_examples = 0
                for batch in val_loader:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    output = self.model(inputs)
                    targets = targets.to(self.device)
                    loss = loss_fn(output, targets)
                    valid_loss += loss.data.item() * inputs.size(0)
                    correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                    num_correct += torch.sum(correct).item()
                    num_examples += correct.shape[0]
                valid_loss /= len(val_loader.dataset)
                valid_losses.append(valid_loss)

                if valid_loss > y_range:
                    y_range = valid_loss

                accuracy =  num_correct / num_examples
                accuracies.append(accuracy)

                print('Эпоха: {}, Ошибка на обучающей выборке: {:.2f}, Ошибка на тестовой выборке: {:.2f}, '
                      'Точность = {:.2f}'.format(epoch, training_loss, valid_loss, accuracy))

                self.train_loss_plot.setData(epochs, training_losses)
                self.valid_loss_plot.setData(epochs, valid_losses)
                self.accuracy_plot.setData(epochs, accuracies)
                self.app.processEvents()
                self.ui.tb_save_model.setEnabled(True)
                self.ui.tb_save_model_as.setEnabled(True)
        except Exception as e:
            self.log(f"Ошибка обучения модели: {e}", QColor(200,0,0))

    @pyqtSlot()
    def on_open_model(self):
        try:
            fname = QFileDialog.getOpenFileName(
                self,
                "Открыть файл параметров обученной модели",
                ".",
                "Параметры обученной модели (*.mdl);; Все файлы (*.*)",
            )
            if fname[0]:
                self.model_filename = fname[0]
                model_file = open(self.model_filename, 'rb')
                self.model = pickle.load(model_file)
                self.model.to(self.device)
                self.ui.tb_save_model.setEnabled(False)
                self.ui.tb_save_model_as.setEnabled(True)
        except Exception as e:
            self.log(f"Ошибка открытия модели: {e}", QColor(200,0,0))

    @pyqtSlot()
    def on_save_model(self):
        if self.model_filename:
            model_file = open(self.model_filename, 'wb')
            pickle.dump(self.model, model_file)
            self.ui.tb_save.setEnabled(False)
        else:
            self.on_save_model_as()

    @pyqtSlot()
    def on_save_model_as(self):
        fname = QFileDialog.getSaveFileName(
            self,
            "Сохранить файл параметров обученной модели",
            ".",
            "Параметры обученной модели (*.mdl);; Все файлы (*.*)",
        )
        if fname[0]:
            self.model_filename = fname[0]
            self.ui.tb_save.setEnabled(False)

    @pyqtSlot()
    def on_open_labels(self):
        fname = QFileDialog.getOpenFileName(
            self,
            "Открыть файл с описанием меток",
            ".",
            "Текст CSV (*.csv);; Все файлы (*.*)",
        )
        if fname[0]:
            if self.labels_data_table_model.open(fname[0]):
                if self.labels_data_table_model.get_type() == LabelsDataTableModel.GESTURES_TYPE:
                    self.mode = TrainWindow.GESTURES_MODE
                else:
                    self.mode = TrainWindow.EMOTIONS_MODE
                self.ui.tv_labels.setModel(self.labels_data_table_model)
                self.ui.cb_labels.setModel(self.labels_data_table_model)
                self.ui.cb_labels.setModelColumn(LabelsDataTableModel.UNICODE_COLUMN)
                self.train_data_model.create(self.labels_data_table_model.get_type())

    @pyqtSlot(str)
    def on_epochs_changed(self, text):
        pass