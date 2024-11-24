from math import isnan

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QSize
import pandas as pd
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QMessageBox


class LabelsDataTableModel(QtCore.QAbstractTableModel):
    ID_COLUMN = 0
    IMAGE_COLUMN = 1
    UNICODE_COLUMN = 2
    NAME_COLUMN = 3

    GESTURES_TYPE = 0
    EMOTIONS_TYPE = 1

    def __init__(self):
        super(LabelsDataTableModel, self).__init__()
        self.noto_font = QFont("Noto Color Emoji", 20)
        self.data = None
        self.type = -1

    def get_type(self):
        return self.type

    def data(self, index, role):
        if role == Qt.ItemDataRole.UserRole:
            return self.data.iloc[index.row(), LabelsDataTableModel.ID_COLUMN]
        elif role == Qt.ItemDataRole.DisplayRole:
            value = self.data.iloc[index.row(), index.column()]
            if value is np.nan:
                return ""
            else:
                return str(value)
        elif role == Qt.ItemDataRole.FontRole and index.column() == self.UNICODE_COLUMN:
            return self.noto_font
        elif role == Qt.ItemDataRole.TextAlignmentRole and index.column() == self.UNICODE_COLUMN:
            return Qt.AlignmentFlag.AlignHCenter

    def rowCount(self, index):
        return self.data.shape[0]

    def columnCount(self, index):
        return self.data.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self.data.columns[section])

    def get_unicode(self, row):
        try:
            value = self.data.iloc[row, self.UNICODE_COLUMN]
            if issubclass(type(value), str):
                return value
            else:
                return None
        except:
            return None

    def get_unicode_by_id(self, id):
        try:
            if self.type == LabelsDataTableModel.EMOTIONS_TYPE:
                df = self.data.set_index('Эмоция')
            else:
                df = self.data.set_index('Жест')
            value = df.loc[id]['Unicode']
            if issubclass(type(value), str):
                return value
            else:
                return None
        except Exception as e:
            print(e)
            return None

    def open(self, filename):
        try:
            df = pd.read_csv(filename)
            if len(df.columns) == 4:
                if df.columns[self.ID_COLUMN] == "Жест":
                    self.type = self.GESTURES_TYPE
                elif df.columns[self.ID_COLUMN] == "Эмоция":
                    self.type = self.EMOTIONS_TYPE
                else:
                    raise Exception("Неподдерживаемый тип меток")
                self.data = df
                self.file_name = filename
                self.modified = False
                self.modelReset.emit()
                return True
            else:
                raise Exception("Недопустимый формат файла")
        except Exception as e:
            dlg = QMessageBox()
            dlg.setWindowTitle("Невозможно загрузить метки!")
            dlg.setText(f"{e}")
            dlg.setStandardButtons(
                QMessageBox.StandardButton.Ok
            )
            dlg.setIcon(QMessageBox.Icon.Critical)
            button = dlg.exec()
            return False