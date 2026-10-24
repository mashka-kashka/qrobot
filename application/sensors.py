from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import numpy as np
import sys
import os
import time
import RPi.GPIO as GPIO
from smbus2 import SMBus
from mlx90614 import MLX90614
import serial

sys.path.insert(0, './DFRobot')
from BloodOxygen_S import DFRobot_BloodOxygen_S_i2c


class QRobotSensors(QObject):
    data_captured_signal = pyqtSignal(object, object, object, object)
    running = False

    def __init__(self):
        super().__init__()
        I2C_1       = 0x01               # I2C_1 Use i2c1 interface (or i2c0 with configuring Raspberry Pi file) to drive sensor
        MAX30102_I2C_ADDRESS = 0x57      # I2C device address, which can be changed by changing A1 and A0, the default address is 0x77
        MLX90614_I2C_ADDRESS = 0x5A
        
        # Датчик пульса и аспирации
        self.max30102 = DFRobot_BloodOxygen_S_i2c(I2C_1, MAX30102_I2C_ADDRESS)
        
        # Датчик температуры пальца
        self.mlx90614 = MLX90614(SMBus(I2C_1), address=MLX90614_I2C_ADDRESS)
        
        # Тепловизор
        self.Tmax = 40
        self.Tmin = 20
        self.mlx90640 = serial.Serial ('/dev/ttyAMA0', 115200, timeout=5)

    @pyqtSlot()
    def start(self):
        self.running = True
        
        while (False == self.max30102.begin()):
            print("init fail!")
            time.sleep(1)
        
        self.max30102.sensor_start_collect()
        time.sleep(1)
        
        self.mlx90640.write(serial.to_bytes([0xA5,0x25,0x01,0xCB]))
        time.sleep(0.1)

        self.mlx90640.write(serial.to_bytes([0xA5,0x35,0x02,0xDC]))
        
        self.get_data() # Получение данных

    def stop(self):
        self.running = False

    @pyqtSlot()
    def get_data(self):
        if not self.running:
            return
            
        data = np.linspace(start=2000, stop=4000, num=768)
            
        try:
            buf = self.mlx90640.read(1544)
            data = np.frombuffer(buf[4:1540], dtype=np.int16)
            norm = np.uint8(data/100)
        except:
            pass
            
        try:
            #norm = np.uint8((data/100 - self.Tmin)*255/(self.Tmax-self.Tmin))
            norm.shape = (24,32)
            self.max30102.get_heartbeat_SPO2()
            self.data_captured_signal.emit(norm, 
                                           self.mlx90614.get_obj_temp(), 
                                           self.max30102.SPO2, 
                                           self.max30102.heartbeat)
        except:
            pass


