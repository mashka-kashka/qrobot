from PyQt6.QtCore import QObject, pyqtSignal
import sounddevice as sd
import vosk
import json
import queue
import toml
import sys


class QRobotVoice(QObject):
    phrase_captured_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.q = queue.Queue() # Хранилище данных с микрофона

        with open('config.toml', 'r') as f:
            self.config = toml.load(f)
            self.microphone = self.config["microphone"]["device"]
            self.model = vosk.Model(self.config["microphone"]["model"])
            self.samplerate = self.config["microphone"]["samplerate"]

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def listen(self):
        dev = sd.query_devices()
        prev_text = None
        with sd.RawInputStream(samplerate=self.samplerate, device=self.microphone,
                               dtype="int16", channels=1, callback=self.callback):
            rec = vosk.KaldiRecognizer(self.model, self.samplerate)
            while True:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())["text"]
                    if res:
                        prev_text = None
                        print(f"Фраза целиком: {res}")
                else:
                    res = json.loads(rec.PartialResult())["partial"]
                    if res and res != prev_text:
                        prev_text = res
                        print(f"Поток: {res}")