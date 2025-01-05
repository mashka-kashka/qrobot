from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtCore import pyqtSlot
import toml
import serial

class QServoController(QObject):

    def __init__(self):
        super().__init__()

        self.controller = None
        try:
            self.controller = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)
        except:
            pass

        if self.controller is None:
            try:
                self.controller = serial.Serial('/dev/ttyACM1', 115200, timeout=0.1)
            except:
                pass # Контроллер не найден - сервоприводы работать не будут

        self.positions = {}
        self.servos = None
        with open('config.toml', 'r') as f:
            self.config = toml.load(f)
            self.servos = self.config["servos"]
            self.period = int(self.config["controller"]["period"])
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.__on_timer)
            self.timer.setInterval(self.period)
            self.timer.start()

    def reset(self):
        command = ''
        for channel in self.servos:
            servo = self.servos[channel]
            command += f"#{channel}P{servo['neutral']}"
        command += 'T0D0\r\n'
        self.__write_command(command)

    def get_servos_count(self):
        return len(self.servos)

    def get_servo_info(self, servo_id):
        channel = None # Канал
        name = None # Название сервопривода
        begin = None # Начальное положение
        end = None # Конечное положение
        neutral = None # Нейтральное положение
        if self.servos:
            channels = list(self.servos.keys())
            channel = channels[servo_id]
            servo = self.servos.get(channel)
            if servo is not None:
                name = servo["name"]
                begin = servo["begin"]
                end = servo["end"]
                neutral = servo["neutral"]

        return channel, name, begin, end, neutral

    def get_servo_range(self, servo_channel):
        begin = None # Начальное положение
        end = None # Конечное положение
        if self.servos:
            servo = self.servos.get(servo_channel)
            if servo is not None:
                begin = servo["begin"]
                end = servo["end"]

        return begin, end

    def set_servo_position(self, channel, position, absolute = False):
        """
        Перемещение сервопривода, подключенного к каналу channel, в позицию position.
        Если параметр absolute = False, то ведётся усреднение позиций, переданных за
        определённый промежуток времени. В противном случае применяется последнее
        переданное значение.
        """
        servo_positions = self.positions.get(channel)
        if servo_positions is None:
            self.positions[channel] = [position]
        else:
            if absolute:
                servo_positions = [position]
            else:
                servo_positions.append(position)

    @pyqtSlot()
    def __on_timer(self):
        command = ''
        for channel, positions in self.positions.items():
            position = int(sum(positions) / len(positions))
            servo = self.servos[str(channel)]
            begin = servo["begin"]
            end = servo["end"]
            if begin > end:
                if position > begin:
                    position = begin
                elif position < end:
                    position = end
            else:
                if position > end:
                    position = end
                elif position < begin:
                    position = begin

            command += f"#{channel}P{position}"

        if command != '':
            command += f"T{int(self.period / 2)}D10\r\n"
            self.__write_command(command)
        self.positions = {}

    def __write_command(self, command):
        if self.controller:
            self.controller.write(command.encode("utf-8"))
            print(f"Выполнение команды: {command}")