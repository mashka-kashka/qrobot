from operator import index
from venv import create

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QModelIndex, QPoint
import pandas as pd
from PyQt6.QtGui import QColor, QPixmap, QPainter, QPen
from PyQt6.QtWidgets import QMessageBox
from google.protobuf.json_format import MessageToDict
from mediapipe.tasks.python.vision import FaceLandmarkerResult
from mediapipe.python.solutions.face_mesh import FACEMESH_NUM_LANDMARKS_WITH_IRISES
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_FACE_OVAL
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_IRISES
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_IRIS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_NOSE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_IRIS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION


class TrainDataTableModel(QtCore.QAbstractTableModel):
    ID_COLUMN = 0

    GESTURES_TYPE = 0
    EMOTIONS_TYPE = 1

    modified = False
    file_name = ''
    white_color = QColor(255, 255, 255)

    def __init__(self):
        super(TrainDataTableModel, self).__init__()
        self.red_pen = QPen()
        self.red_pen.setWidth(1)
        self.red_pen.setColor(QColor(200, 0, 0))
        self.green_pen = QPen()
        self.green_pen.setWidth(1)
        self.green_pen.setColor(QColor(0, 200, 0))
        self.df = pd.DataFrame()

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if role == Qt.ItemDataRole.EditRole:
            try:
                self.df.iat[index.row(), index.column()] = value
                return True
            except:
                return False
        return False

    def data(self, index, role):
        if role == Qt.ItemDataRole.DecorationRole:
            if index.column() == 0:
                if self.type == TrainDataTableModel.EMOTIONS_TYPE:
                    return self.draw_face(self.df.index[index.row()])
                else:
                    return self.draw_palm(self.df.index[index.row()])
        elif role == Qt.ItemDataRole.DisplayRole:
                value = self.df.iloc[index.row(), index.column()]
                if value is np.nan:
                    return ""
                else:
                    return str(value)

    def draw_face(self, row_index, size = 27):
        pixmap = QPixmap(size, size)
        pixmap.fill(self.white_color)
        try:
            row = self.df.loc[row_index, :]
            painter = QPainter(pixmap)
            painter.setPen(self.red_pen)
            points = []
            for i in range(FACEMESH_NUM_LANDMARKS_WITH_IRISES):
                x = int(row.iloc[i * 3 + 1] * size)
                y = int(row.iloc[i * 3 + 2] * size)
                points.append(QPoint(x, y))

            FACEMESH = frozenset().union(*[
                FACEMESH_RIGHT_IRIS, FACEMESH_LEFT_IRIS, FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW,
                FACEMESH_RIGHT_EYE, FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_NOSE
            ])

            painter.setPen(self.green_pen)
            for start, end in FACEMESH:
                painter.drawLine(points[start], points[end])

            painter.end()
        except Exception as e:
            pass

        return pixmap

    def draw_palm(self, row_index, size = 27):
        pixmap = QPixmap(size, size)
        pixmap.fill(self.white_color)
        try:
            row = self.df.loc[row_index, :]
            painter = QPainter(pixmap)
            painter.setPen(self.red_pen)
            points = []
            for i in range(len(self.landmarks)):
                x = int(row.iloc[i * 3 + 2] * size)
                y = int(row.iloc[i * 3 + 3] * size)
                points.append(QPoint(x, y))
                painter.drawEllipse(x - 1, y - 1, 2, 2)

            painter.setPen(self.green_pen)
            segments = [[0,1,2,3,4],[0,5,6,7,8],[0,17,18,19,20],[9,10,11,12],[13,14,15,16],[5,9,13,17]]
            for segment in segments:
                for i in range(1, len(segment)):
                    painter.drawLine(points[segment[i - 1]], points[segment[i]])
            painter.end()

        except Exception as e:
            pass
        return pixmap

    def rowCount(self, index):
        return self.df.shape[0]

    def columnCount(self, index):
        return self.df.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self.df.columns[section])

    def is_modified(self):
        return self.modified

    def get_file_name(self):
        return  self.file_name

    def create(self, type):
        self.type = type
        if type == self.GESTURES_TYPE:
            _data = {
                    "Жест": pd.Series([], dtype=np.dtype("int8")),
                    "Уверенность": pd.Series([], dtype=np.float64), # confidence
                    }

            self.landmarks = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP',
                  'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP',
                  'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
                  'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
            for i, landmark in enumerate(self.landmarks):
                for axis in ['X_', 'Y_', 'Z_']:
                    _data[axis + landmark] = pd.Series([], dtype=np.float64)
        elif type == self.EMOTIONS_TYPE:
            _data = {
                    "Эмоция": pd.Series([], dtype=np.dtype("int8")),
                    }

            self.landmarks = []
            for i in range(FACEMESH_NUM_LANDMARKS_WITH_IRISES):
                for axis in ['X_', 'Y_', 'Z_']:
                    _data[axis + str(i)] = pd.Series([], dtype=np.float64)

            self.blendshapes = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft',
                                'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
                                'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
                                'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
                                'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
                                'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
                                'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft',
                                'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft',
                                'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker',
                                'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
                                'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
                                'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']
            for blendshape in self.blendshapes:
                _data[blendshape] = pd.Series([], dtype=np.float64)
        else:
            return

        self.df = pd.DataFrame(_data)
        self.modified = False
        self.modelReset.emit()

    def get_sample(self, results, X = False):
        if not results:
            return None

        try:
            sample = []
            if self.type == TrainDataTableModel.EMOTIONS_TYPE : # Лицо
                if not results.face_landmarks:
                    return None

                landmark = results.face_landmarks[0]
            else: # Руки
                if not results.multi_hand_landmarks:
                    return None

                score = MessageToDict(results.multi_handedness[0])['classification'][0]['score']
                sample.append(score)
                landmark = results.multi_hand_landmarks[0].landmark

            min_x = None
            min_y = None
            min_z = None
            max_x = None
            max_y = None
            max_z = None

            for lm in landmark:
                if not min_x or lm.x < min_x:
                    min_x = lm.x
                if not min_y or lm.y < min_y:
                    min_y = lm.y
                if not min_z or lm.z < min_z:
                    min_z = lm.z
                if not max_x or lm.x > max_x:
                    max_x = lm.x
                if not max_y or lm.y > max_y:
                    max_y = lm.y
                if not max_z or lm.z > max_z:
                    max_z = lm.z

            dx = max_x - min_x
            dy = max_y - min_y
            dz = max_z - min_z

            scale = max(dx, dy, dz)

            for i,lm in enumerate(landmark):
                sample.append((lm.x - min_x - dx / 2.) / scale + 0.5)
                sample.append((lm.y - min_y - dy / 2.) / scale + 0.5)
                sample.append((lm.z - min_z - dz / 2.) / scale + 0.5)

            if self.type == TrainDataTableModel.EMOTIONS_TYPE : # Лицо
                for i in range(len(self.blendshapes)):
                    sample.append(results.face_blendshapes[0][i].score)

            if X:
                if self.type == TrainDataTableModel.EMOTIONS_TYPE:
                    return sample[-len(self.blendshapes):]
                else:
                    return sample
            else:
                return sample
        except Exception as e:
            print(f"{e}")
        return None

    def add(self, sample_id, results):
        sample = self.get_sample(results)
        if not sample:
            return

        sample.insert(0, sample_id)
        df = pd.DataFrame([sample], columns=self.df.columns)
        self.df = pd.concat([self.df, df], ignore_index=True)
        self.modified = True
        self.modelReset.emit()

    def centering(self):
        self.df.reset_index(inplace=True)
        for row_index, row in self.df.iterrows():
            min_x = None
            min_y = None
            min_z = None
            max_x = None
            max_y = None
            max_z = None

            for i in range(len(self.landmarks)):
                x = row.iloc[i * 3 + 2]
                y = row.iloc[i * 3 + 3]
                z = row.iloc[i * 3 + 4]
                if not min_x or x < min_x:
                    min_x = x
                if not min_y or y < min_y:
                    min_y = y
                if not min_z or z < min_z:
                    min_z = z
                if not max_x or x > max_x:
                    max_x = x
                if not max_y or y > max_y:
                    max_y = y
                if not max_z or z > max_z:
                    max_z = z

            dx = 0.5 - min_x - (max_x - min_x) * 0.5
            dy = 0.5 - min_x - (max_y - min_y) * 0.5
            dz = 0.5 - min_x - (max_z - min_z) * 0.5

            for i in range(len(self.landmarks)):
                row.iat[i * 3 + 2] += dx
                row.iat[i * 3 + 3] += dy
                row.iat[i * 3 + 4] += dz

    def open(self, filename, centering = False):
        try:
            df = pd.read_csv(filename)
            if (df.columns == self.df.columns).all():
                self.df = df
                if (centering):
                    self.centering()
                self.file_name = filename
                self.modified = False
                self.modelReset.emit()
        except Exception as e:
            dlg = QMessageBox()
            dlg.setWindowTitle("Невозможно загрузить данные!")
            dlg.setText(f"{e}")
            dlg.setStandardButtons(
                QMessageBox.StandardButton.Ok
            )
            dlg.setIcon(QMessageBox.Icon.Critical)
            button = dlg.exec()

    def save(self, filename):
        self.file_name = filename
        self.df.to_csv(filename, index=False)
        self.modified = False
        self.modelReset.emit()

    def removeRow(self, row):
        self.beginRemoveRows(QModelIndex(), row, row)
        self.df.drop([self.df.index[row]], inplace=True)
        self.endRemoveRows()

    def X(self):
        if self.type == TrainDataTableModel.EMOTIONS_TYPE:
            return self.df.loc[:, '_neutral':]
        else:
            return self.df.loc[:,'Уверенность':]

    def y(self):
        if self.type == TrainDataTableModel.EMOTIONS_TYPE:
            labels = self.df['Эмоция']
        else:
            labels = self.df['Жест']

        # Замена идентификаторов на их порядковые номера
        replacement = {}
        for index, id in enumerate(labels.unique()):
            replacement[id] = index

        return labels.replace(replacement)

    def get_labels(self):
        if self.type == TrainDataTableModel.EMOTIONS_TYPE:
            return self.df['Эмоция'].unique()
        else:
            return self.df['Жест'].unique()
