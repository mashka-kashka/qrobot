import math
import toml
from PyQt6.QtCore import QObject, QPoint
from PyQt6.QtGui import QFont, QImage, QPainter, QPen, QColor
from google.protobuf.json_format import MessageToDict
import mediapipe as mp
from mediapipe import solutions
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import hands as mp_hand_detector
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import pickle
import torch
import os
import random


class QRobot(QObject):

    DEFAULT_MODE = 0
    GAME_MODE = 1

    FACE_BLENDSHAPES = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft',
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
    HAND_LANDMARKS = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP',
                      'INDEX_FINGER_PIP',
                      'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP',
                      'MIDDLE_FINGER_DIP',
                      'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
                      'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
    HAND_TIPS =      [('THUMB_TIP','PINKY_MCP'),
                      ('INDEX_FINGER_TIP', 'INDEX_FINGER_MCP'),
                      ('MIDDLE_FINGER_TIP', 'MIDDLE_FINGER_MCP'),
                      ('RING_FINGER_TIP', 'RING_FINGER_MCP'),
                      ('PINKY_TIP', 'PINKY_MCP')]
    SERVOS_INDEX =   { 'LEFT_THUMB_TIP': 8, 'LEFT_INDEX_FINGER_TIP': 9, 'LEFT_MIDDLE_FINGER_TIP': 10,
                       'LEFT_RING_FINGER_TIP': 12, 'LEFT_PINKY_TIP': 11, 'RIGHT_THUMB_TIP': 25,
                       'RIGHT_INDEX_FINGER_TIP': 24, 'RIGHT_MIDDLE_FINGER_TIP': 22, 'RIGHT_RING_FINGER_TIP': 23,
                       'RIGHT_PINKY_TIP': 21}
    POSE_LANDMARKS = [ 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_HIP', 'RIGHT_HIP',
                       'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
                       'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    POSE_LANDMARK_IDS = list(range(11, 15)) + list(range(23, 33))
    ROBOT_SEGMENTS = ([[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 17, 18, 19, 20], [9, 10, 11, 12], [13, 14, 15, 16],
                      [5, 9, 13, 17], [21, 22, 23, 24, 25], [21, 26, 27, 28, 29], [21, 38, 39, 40, 41],
                      [30, 31, 32, 33], [34, 35, 36, 37], [26, 30, 34, 38], [0, 44, 42, 43, 45, 21],
                      [43, 47, 49, 51, 53, 55, 51], [42, 46, 48, 50, 52, 54, 50], [46, 47]])
    ARM_PREFIXES = ['LEFT_', 'RIGHT_']

    def __init__(self, controller):
        super().__init__()

        self.controller = controller
        self.mode = self.DEFAULT_MODE

        self.red_pen = QPen()
        self.red_pen.setWidth(3)
        self.red_pen.setColor(QColor(200, 0, 0))
        self.green_pen = QPen()
        self.green_pen.setWidth(3)
        self.green_pen.setColor(QColor(0, 200, 0))
        self.emoji_font = QFont("Noto Color Emoji", 64)
        self.robot_data = {}
        self.servo_values = np.zeros(32)

        # Список наименований ключевых точек
        QRobot.ROBOT_LANDMARKS = []
        for prefix in QRobot.ARM_PREFIXES:
            self.ROBOT_LANDMARKS += [prefix + x for x in QRobot.HAND_LANDMARKS]
        QRobot.ROBOT_LANDMARKS += QRobot.POSE_LANDMARKS

        # Модуль распознавания ладони на изображении
        self.hand_detector = mp_hand_detector.Hands(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2)

        # Модуль распознавания поз на изображении
        # base_options = python.BaseOptions(model_asset_path='../models/pose_landmarker_full.task')
        # options = vision.PoseLandmarkerOptions(
        #     base_options=base_options,
        #     output_segmentation_masks=True,
        #     num_poses=1)
        # self.pose_detector = vision.PoseLandmarker.create_from_options(options)

        # Модуль распознавания лиц на изображении
        base_options = python.BaseOptions(model_asset_path='../models/face_landmarker.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                              output_face_blendshapes=True,
                                              output_facial_transformation_matrixes=True,
                                              num_faces=1)
        self.face_detector = vision.FaceLandmarker.create_from_options(options)

        # Модели для распознвания жестов и эмоций
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            torch.serialization.register_package(0, lambda x: x.device.type, lambda x, _: x.cpu())

        self.gestures_labels = None
        self.emotions_labels = None
        self.gestures_model = None
        self.emotions_model = None

        with open('config.toml', 'r') as f:
            self.config = toml.load(f)
            models = self.config["models"]
            gestures_labels_filename = f'{os.getcwd()}/{models["gestures_labels"]}'
            self.gestures_labels = pd.read_csv(gestures_labels_filename)

            gestures_model_filename = f'{os.getcwd()}/{models["gestures_model"]}'
            model_file = open(gestures_model_filename, 'rb')
            self.gestures_model = pickle.load(model_file)
            self.gestures_model.to(self.device)

            emotions_labels_filename = f'{os.getcwd()}/{models["emotions_labels"]}'
            self.emotions_labels = pd.read_csv(emotions_labels_filename)

            emotions_model_filename = f'{os.getcwd()}/{models["emotions_model"]}'
            model_file = open(emotions_model_filename, 'rb')
            self.emotions_model = pickle.load(model_file)
            self.emotions_model.to(self.device)

    def get_ranges(self, input_list, width, height):
        x_min = x_max = None
        y_min = y_max = None
        z_min = z_max = None
        visible = False
        data = None
        if input_list:
            for lm in input_list:
                if not x_min or lm.x < x_min:
                    x_min = max(0.0, lm.x)
                if not y_min or lm.y < y_min:
                    y_min = max(0.0, lm.y)
                if not z_min or lm.z < z_min:
                    z_min = max(0.0, lm.z)
                if not x_max or lm.x > x_max:
                    x_max = min(1.0, lm.x)
                if not y_max or lm.y > y_max:
                    y_max = min(1.0, lm.y)
                if not z_max or lm.z > z_max:
                    z_max = min(1.0, lm.z)
                if x_min > 0.0 or x_max < 1.0 or y_min > 0.0 or y_max < 1.0:
                    visible = True
            data = {'visible': visible,
                    'x_min': x_min, 'x_max': x_max, 'dx': x_max - x_min,
                    'y_min': y_min, 'y_max': y_max, 'dy': y_max - y_min,
                    'z_min': z_min, 'z_max': z_max, 'dz': z_max - z_min,
                    'rect': (int(x_min * width), int(y_min * height), int(x_max * width), int(y_max * height))}
        return data
    # Обработка кадра
    def process_frame(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        height = mp_image.height
        width = mp_image.width

        data = {'Кадр' : {'Ширина' : width, 'Высота' : height}}

        emotion = 0
        # Распознавание лиц
        face_detection_result = self.face_detector.detect(mp_image)
        annotated_image = self.draw_faces_on_image(image, face_detection_result)

        if len(face_detection_result.face_landmarks) > 0:
            ranges = self.get_ranges(face_detection_result.face_landmarks[0], width, height)
            data['Лицо'] = ranges
            emotion = self.detect_emotion(face_detection_result.face_blendshapes[0])
            if not emotion is None:
                emotion = (int(emotion),self.emotions_labels.loc[emotion]['Unicode'])
                data['Лицо']['Эмоция'] = emotion

        # Распознавание рук и поз

        right_gesture = 0
        left_gesture = 0

        hand_detection_results = self.hand_detector.process(image)
        hand_landmarks_list = hand_detection_results.multi_hand_landmarks
        if hand_landmarks_list:
            for idx, lm in enumerate(hand_landmarks_list):
                bounds = self.get_ranges(lm.landmark, width, height)
                classification = hand_detection_results.multi_handedness[idx].classification[0]
                palm = "Правая ладонь" if classification.label == 'Left' else "Левая ладонь"
                data[palm] = bounds
                score = MessageToDict(hand_detection_results.multi_handedness[idx])['classification'][0]['score']
                gesture = self.detect_gesture(lm.landmark, bounds, score)
                if not gesture is None:
                    data[palm]['Жест'] = (int(gesture),self.gestures_labels.loc[gesture]['Unicode'])
                    if classification.label == 'Left':
                        left_gesture = int(gesture)
                    else:
                        right_gesture = int(gesture)
                    #print(f"{palm}: {gesture}")

        sceleton = {}
        # pose_detection_result = self.pose_detector.detect(mp_image)
        # pose_landmarks_list = pose_detection_result.pose_landmarks
        # if pose_landmarks_list:
        #     for i, idx in enumerate(QRobot.POSE_LANDMARK_IDS):
        #         lm = pose_landmarks_list[0][idx]
        #         #if lm.visibility > 0.9:
        #         sceleton[QRobot.POSE_LANDMARKS[i]] = {'x': lm.x, 'y': lm.y, 'z': lm.z,
        #                                                   'point': (int(lm.x * width), int(lm.y * height))}
        if hand_landmarks_list:
            for idx, lm in enumerate(hand_landmarks_list):
                is_right_palm = hand_detection_results.multi_handedness[idx].classification[0].label == 'Left'
                for pidx, pt in enumerate(lm.landmark):
                    if is_right_palm:
                        sceleton['RIGHT_' + QRobot.HAND_LANDMARKS[pidx]] = {'x': pt.x, 'y': pt.y, 'z': pt.z,
                                                                            'point': (int(pt.x * width),
                                                                                      int(pt.y * height))}
                    else:
                        sceleton['LEFT_' + QRobot.HAND_LANDMARKS[pidx]] = {'x': pt.x, 'y': pt.y, 'z': pt.z,
                                                                           'point': (int(pt.x * width),
                                                                                     int(pt.y * height))}
        if self.mode == self.GAME_MODE:
            if right_gesture == 1 and left_gesture == 1:  # ✋ + ✋
                print(f"Сброс")
                self.mode = self.DEFAULT_MODE
                self.controller.set_servo_position(19, 0)
                self.controller.set_servo_position(18, 1500)
                self.controller.set_servo_position(20, 0)
            else:
                if emotion == 12: # 🙁,Слегка нахмуренное лицо
                    if left_gesture == 1:  # Бумага
                        self.show_rock()
                    elif left_gesture == 7:  # Ножницы
                        self.show_paper()
                    elif left_gesture == 26:  # Камень
                        self.show_scissors()
                else:
                    random_num = random.randint(0, 3)

                    if random_num == 0:    # Бумага
                        while True:
                            self.show_rock()
                            break
                    elif random_num == 1:  # Ножницы
                        while True:
                            self.show_paper()
                            break
                    elif random_num == 2:  # Камень
                        while True:
                            self.show_scissors()
                            break

        else:
            if right_gesture == 7 and left_gesture == 7:  # ✌ + ✌
                print(f"Переключение в режим игры")
                self.mode = self.GAME_MODE
                self.controller.set_servo_position(19, 1700)
                self.controller.set_servo_position(18, 2500)
                self.controller.set_servo_position(20, 1800)
            else:
                # Расчеты по скелету
                self.calc_sceleton(sceleton)

        data['Скелет'] = sceleton
#        with open("sceleton.json", "w") as outfile:
#            json.dump(data, outfile)

        annotated_image = self.draw_sceleton_on_image(annotated_image, data)
        annotated_image = self.draw_emotion_on_image(annotated_image, data)
        annotated_image = self.draw_gestures_on_image(annotated_image, data)

        return annotated_image, data

    def show_rock(self):
        self.controller.set_servo_position(25, 1200)
        self.controller.set_servo_position(24, 1700)
        self.controller.set_servo_position(22, 1650)
        self.controller.set_servo_position(23, 1200)
        self.controller.set_servo_position(21, 1250)

    def show_scissors(self):
        self.controller.set_servo_position(25, 1200)
        self.controller.set_servo_position(24, 1100)
        self.controller.set_servo_position(22, 1200)
        self.controller.set_servo_position(23, 1200)
        self.controller.set_servo_position(21, 1250)

    def show_paper(self):
        self.controller.set_servo_position(25, 1200)
        self.controller.set_servo_position(24, 1100)
        self.controller.set_servo_position(22, 1200)
        self.controller.set_servo_position(23, 1500)
        self.controller.set_servo_position(21, 1500)

    def calc_sceleton(self, sceleton):

        # Расчёт положений пальцев для управления сервоприводами
        for prefix in ['LEFT_', 'RIGHT_']:
            wrist = sceleton.get(prefix + 'WRIST')
            if wrist is None:
                continue

            vwrist = np.array([[wrist['x'], wrist['y'], wrist['z']]])
            for tip_idx, mcp_idx in self.HAND_TIPS:
                tip_name = prefix + tip_idx
                mcp_name = prefix + mcp_idx

                # Расчёт расстояния от кончика пальца до основания ладони
                tip = sceleton.get(tip_name)
                if tip is None:
                    continue
                vtip = np.array([[tip['x'], tip['y'], tip['z']]])
                v = vtip - vwrist
                tip_dist = np.linalg.norm(v)

                # Расчёт расстояния от основания пальца до основания ладони
                mcp = sceleton.get(mcp_name)
                if mcp is None:
                    continue
                vmcp = np.array([[mcp['x'], mcp['y'], mcp['z']]])
                v = vmcp - vwrist
                mcp_dist = np.linalg.norm(v)

                servo_channel = self.SERVOS_INDEX.get(tip_name)
                if servo_channel is not None:

                    begin, end = self.controller.get_servo_range(str(servo_channel))
                    val = tip_dist / mcp_dist
                    # Если tip_dist / mcp_dist больше 1, то палец - разогнут, иначе - согнут.
                    self.controller.set_servo_position(servo_channel, begin if val > 1 else end, True)

        # Расчёты углов
        # ls = sceleton.get('LEFT_SHOULDER')
        # lh = sceleton.get('LEFT_HIP')
        # le = sceleton.get('LEFT_ELBOW')
        # lw = sceleton.get('LEFT_WRIST')
        # rs = sceleton.get('RIGHT_SHOULDER')
        # rh = sceleton.get('RIGHT_HIP')
        # re = sceleton.get('RIGHT_ELBOW')
        # rw = sceleton.get('RIGHT_WRIST')
        #
        # while (ls is not None) and (rs is not None): # На кадре видно и левое, и правое плечо
        #     vls = np.array([[ls['x'], ls['y'], ls['z']]]) # Левое плечо
        #     vrs = np.array([[rs['x'], rs['y'], rs['z']]]) # Правое плечо
        #     vlrs = self.normalize(vrs - vls) # Нормализованный вектор, направленный от левого плеча к правому
        #     vrls = vlrs * -1  # Нормализованный вектор, направленный от правого плеча к левому
        #
        #     if rh is not None: # Видно правое бедро
        #         vrsh = np.array([[rh['x'], rh['y'], rh['z']]]) - vrs # Вектор, направленный от правого плеча к бедру
        #         vc = self.normalize(np.cross(vlrs, vrsh)) # Нормаль к плоскости торса
        #         vspine = self.normalize(np.cross(vlrs,vc)) # Вектор по направлению позвоночника
        #     elif lh is not None: # Видно левое бедро
        #         vlsh = np.array([[lh['x'], lh['y'], lh['z']]]) - vls # Вектор, направленный от левого плеча к бедру
        #         vc = self.normalize(np.cross(vlsh, vlrs))  # Нормаль к плоскости торса
        #     else:
        #         break
        #
        #     if le is not None: # Видно левый локоть
        #         vle = np.array([[le['x'], le['y'], le['z']]]) # Левый локоть
        #         vlse = self.normalize(vle - vls) # Нормализованный вектор, направленный от левого плеча к локтю
        #         vlsec = np.dot(vlse, vc.reshape(3,1)) * vc  # Проекция на переднюю плоскость туловища
        #         vlv = self.normalize(vlse - vlsec)
        #         plsess = np.dot(vlv, vrls.reshape(3,1))  # Произведение на вектор между плечами
        #         alsess = int(np.rad2deg(np.arccos(plsess))[0][0] / 20) * 20
        #
        #         # Подъём левого плеча
        #         servo_channel = '14'
        #         servo = self.config["servos"][servo_channel]
        #         mid_val = (servo['max'] - servo['min']) / 2
        #
        #         up_direction = np.dot(vlv, vspine.reshape(3,1)) > 0
        #
        #         if up_direction:  # Направление вверх
        #             val = (90 - alsess) / 90. * mid_val + alsess / 90. * servo['min']
        #         else:
        #             val = (90 - alsess) / 90. * mid_val + alsess / 90. * servo['max']
        #
        #         val = int(val / 100) * 100
        #
        #         if self.servo_values[int(servo_channel)] != val:
        #             print(f'Подъём левого плеча: {int(alsess)} {val}')
        #             self.servo_values[int(servo_channel)] = val
        #             #command += f"#{servo_channel}P{val}"
        #
        #         vlselrs = np.dot(vlse, vlrs.reshape(3,1)) * vlrs  # Проекция на боковую плоскость туловища
        #         vlc = self.normalize(vlse - vlselrs)
        #         plc = np.dot(vlc, vc.reshape(3,1))  # Произведение на нормаль к торсу
        #         lc_direction = 'вверх' if np.dot(vlc, vspine.reshape(3,1)) > 0 else 'вниз'  # Направление
        #         alс = np.rad2deg(np.arccos(plc))
        #         #print(f'Левый угол с торсом: {alс} {lc_direction}')
        #
        #         if lw is not None:  # Видно левую ладонь
        #             vls = np.array([[ls['x'], ls['y'], 0]])  # Левое плечо
        #             vle = np.array([[le['x'], le['y'], 0]])  # Левый локоть
        #             vlse = self.normalize(vle - vls)  # Нормализованный вектор, направленный от левого плеча к локтю
        #             vlw = np.array([[lw['x'], lw['y'], 0]])  # Левая ладонь
        #             vlwe = self.normalize(vlw - vle)  # Нормализованный вектор, направленный от левой ладони к локтю
        #             plwe = np.dot(vlse, vlwe.reshape(3, 1))
        #             alwe = np.rad2deg(np.arccos(plwe))
        #             #print(f'Левый угол в локте: {alwe}')
        #
        #     if re is not None: # Видно правый локоть
        #         vre = np.array([[re['x'], re['y'], re['z']]])  # Правый локоть
        #         vrse = self.normalize(vre - vrs) # Нормализованный вектор, направленный от правого плеча к локтю
        #         vrsec = np.dot(vrse, vc.reshape(3,1)) * vc  # Проекция на переднюю плоскость туловища
        #         vrv = self.normalize(vrse - vrsec)
        #         prsess = np.dot(vrv, vlrs.reshape(3,1))  # Произведение на вектор между плечами
        #         rsess_direction = 'вверх' if np.dot(vrv, vspine.reshape(3,1)) > 0 else 'вниз'  # Направление
        #         arsess = np.rad2deg(np.arccos(prsess))
        #
        #         vrserss = np.dot(vrse, vrls.reshape(3,1)) * vrls  # Проекция на боковую плоскость туловища
        #         vrc = self.normalize(vrse - vrserss)
        #         prc = np.dot(vrc, vc.reshape(3,1))  # Произведение на нормаль к торсу
        #         rc_direction = 'вверх' if np.dot(vrc, vspine.reshape(3,1)) > 0 else 'вниз'  # Направление
        #         arseс = np.rad2deg(np.arccos(prc))
        #
        #         #print(f'Правый угол с торсом: {arseс} {rc_direction} Правый угол с плечами: {arsess} {rsess_direction}')
        #
        #         if rw is not None:  # Видно правую ладонь
        #             vrs = np.array([[rs['x'], rs['y'], 0]])  # Правое плечо
        #             vre = np.array([[re['x'], re['y'], 0]])  # Правый локоть
        #             vrse = self.normalize(vre - vrs)  # Нормализованный вектор, направленный от правого плеча к локтю
        #             vrw = np.array([[rw['x'], rw['y'], 0]])  # Правая ладонь
        #             vrwe = self.normalize(vrw - vre)  # Нормализованный вектор, направленный от правого ладони к локтю
        #             prwe = np.dot(vrse, vrwe.reshape(3, 1))
        #             arwe = np.rad2deg(np.arccos(prwe))
        #             #print(f'Правый угол в локте: {arwe}')
        #
        #     break
        #
        # if command != '':
        #     command += 'T10D10\r\n'

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def detect_gesture(self, landmarks, ranges, score):
        if not self.emotions_model:
            return None

        sample = [score]

        dx = ranges['dx']
        dy = ranges['dy']
        dz = ranges['dz']
        min_x = ranges['x_min']
        min_y = ranges['y_min']
        min_z = ranges['z_min']
        scale = max(dx, dy, dz)

        for i, lm in enumerate(landmarks):
            sample.append((lm.x - min_x - dx / 2.) / scale + 0.5)
            sample.append((lm.y - min_y - dy / 2.) / scale + 0.5)
            sample.append((lm.z - min_z - dz / 2.) / scale + 0.5)

        input = torch.unsqueeze(torch.tensor(sample).double(), dim=0).to(self.device)
        prediction = self.gestures_model(input)
        score = max(prediction[0])
        return self.gestures_model.get_label(prediction)

    def draw_gestures_on_image(self, image, data):
        if self.gestures_labels is None:
            return image

        painter = QPainter(image)
        painter.setFont(self.emoji_font)

        for palm in ['Правая ладонь', 'Левая ладонь']:
            try:
                label = data[palm]['Жест'][1]
                if issubclass(type(label), str):
                    pos_x = 5

                    if palm == 'Левая ладонь':
                        pos_x = image.width() - 100

                    painter.drawText(QPoint(pos_x, image.height() - 30), label)
            except:
                pass

        painter.end()
        return image

    def detect_emotion(self, face_blendshapes):
        if self.emotions_model is None:
            return None

        sample = []
        for i in range(len(QRobot.FACE_BLENDSHAPES)):
            sample.append(face_blendshapes[i].score)

        input = torch.unsqueeze(torch.tensor(sample).double(), dim=0).to(self.device)
        prediction = self.emotions_model(input)
        score = max(prediction[0])
        return self.emotions_model.get_label(prediction)

    def draw_emotion_on_image(self, image, data):
        if self.emotions_labels is None:
            return image

        try:
            label = data['Лицо']['Эмоция'][1]
            if issubclass(type(label), str):
                painter = QPainter(image)
                painter.setFont(self.emoji_font)
                painter.drawText(QPoint(5, 75), label)
                painter.end()
        except:
            pass

        return image

    def draw_sceleton_on_image(self, image, data):
        painter = QPainter(image)

        sceleton = data['Скелет']
        painter.setPen(self.green_pen)
        for segment in QRobot.ROBOT_SEGMENTS:
            for i in range(1, len(segment)):
                start_lm = self.ROBOT_LANDMARKS[segment[i - 1]]
                if not start_lm in sceleton:
                    continue

                end_lm = self.ROBOT_LANDMARKS[segment[i]]
                if not end_lm in sceleton:
                    continue

                a = sceleton[start_lm]['point']
                b = sceleton[end_lm]['point']
                painter.drawLine(a[0], a[1], b[0], b[1])
        painter.end()
        return image

    def draw_poses_on_image(self, rgb_image, pose_detection_result, hand_detection_results):
        height = rgb_image.height()
        width = rgb_image.width()

        pose_landmarks_list = pose_detection_result.pose_landmarks
        hand_landmarks_list = hand_detection_results.multi_hand_landmarks
        wrists = []
        if hand_landmarks_list:
            for lm in hand_landmarks_list:
                wrist = lm.landmark[mp_hands.HandLandmark.WRIST]
                wrists.append((wrist.x, wrist.y, wrist.z))

        landmarks = []
        if pose_landmarks_list:
            for landmark in pose_landmarks_list:
                for idx in self.POSE_LANDMARK_IDS:
                    lm = landmark[idx]
                    point = (lm.x, lm.y, lm.z)
                    if idx == 15 or idx == 16: # Запястье - нужно заменить на координаты руки
                        min_dist = 0.01
                        nearest_wrist_idx = None
                        for wrist_idx, wrist in enumerate(wrists):
                            distance = self.get_distance(point, wrist)
                            if distance < min_dist:
                                min_dist = distance
                                nearest_wrist_idx = wrist_idx
                        if nearest_wrist_idx:
                            landmark.append(wrists[nearest_wrist_idx])
                        else:
                            landmarks.append(point)
                    else:
                        landmarks.append(point)

            painter = QPainter(rgb_image)
            painter.setPen(self.red_pen)
            points = []
            for lm in landmarks:
                x = int(lm[0] * width)
                if x < 0:
                    x = 0
                if x > width:
                    x = width
                y = int(lm[1] * height)
                if y < 0:
                    y = 0
                if y > height:
                    y = height
                points.append(QPoint(x, y))
                painter.drawEllipse(x - 1, y - 1, 2, 2)

            painter.setPen(self.green_pen)
            for segment in self.ROBOT_SEGMENTS:
                for i in range(1, len(segment)):
                    painter.drawLine(points[segment[i - 1]], points[segment[i]])
            painter.end()

        return rgb_image

    def get_distance(self, a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def draw_hands_on_image(self, rgb_image, detection_result):
        annotated_image = np.copy(rgb_image)
        hand_landmarks_list = detection_result.multi_hand_landmarks
        if hand_landmarks_list:
            for handLandmark in hand_landmarks_list:
                mp_drawing.draw_landmarks(annotated_image, handLandmark,
                                          mp_hand_detector.HAND_CONNECTIONS)
        return annotated_image

    def draw_faces_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            # solutions.drawing_utils.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks_proto,
            #     connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp.solutions.drawing_styles
            #     .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        image = QImage(
            annotated_image.data,
            annotated_image.shape[1],
            annotated_image.shape[0],
            QImage.Format.Format_BGR888,
        )

        painter = QPainter(image)
        painter.setFont(self.emoji_font)
        #painter.drawText(QPoint(5, 75), "😀")
        painter.end()

        return image
