from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
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

    @pyqtSlot()
    def listen(self):
        dev = sd.query_devices()
        self.prev_phrase = None
        with sd.RawInputStream(samplerate=self.samplerate, device=self.microphone,
                               dtype="int16", channels=1, callback=self.callback):
            rec = vosk.KaldiRecognizer(self.model, self.samplerate)
            while True:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    phrase = json.loads(rec.Result())["text"]
                    if phrase:
                        self.prev_phrase = None
                        print(f"Фраза: {phrase}")
                        self.phrase_captured_signal.emit(phrase)
                else:
                    phrase = json.loads(rec.PartialResult())["partial"]
                    if phrase and phrase != self.prev_phrase:
                        self.prev_phrase = phrase
                        print(f"Отрывок фразы: {phrase}")