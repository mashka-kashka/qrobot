from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import sounddevice as sd
from fuzzywuzzy import fuzz
import vosk
import json
import queue
import toml
import torch
import sys
import time


class QRobotVoice(QObject):
    phrase_captured_signal = pyqtSignal(str)
    command_recognized_signal = pyqtSignal(str, str)
    mute_mic = False

    def __init__(self):
        super().__init__()

        self.q = queue.Queue() # Хранилище данных с микрофона

        with open('config.toml', 'r') as f:
            self.config = toml.load(f)
            self.model = vosk.Model(self.config["microphone"]["model"])
            self.samplerate = self.config["microphone"]["samplerate"]
            self.min_command_confidence = self.config["microphone"]["min_command_confidence"]

            # Модели для синтеза голоса
            tts = self.config["tts"]
            self.sample_rate = tts["sample_rate"]
            self.speaker = tts["speaker"]
            #torch.set_num_threads(8)  # количество задействованных потоков CPU

            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            self.tts_model = torch.package.PackageImporter("../models/v4_ru.pt").load_pickle("tts_models", "model")
            torch._C._jit_set_profiling_mode(False)
            torch.set_grad_enabled(False)
            self.tts_model.to(self.device)

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if self.mute_mic:
            return
        self.q.put(bytes(indata))

    @pyqtSlot()
    def listen(self):
        dev = sd.query_devices()
        self.prev_phrase = None
        with sd.RawInputStream(samplerate=self.samplerate, device=len(dev) - 1,
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
                        self.recognize_command(phrase)
                else:
                    phrase = json.loads(rec.PartialResult())["partial"]
                    if phrase and phrase != self.prev_phrase:
                        self.prev_phrase = phrase
                        print(f"Отрывок фразы: {phrase}")

    def recognize_command(self, cmd):
        max_ratio = 0
        res_command = None
        commands = self.config["command"]
        for command in commands:
            activation_phrases = commands[command]["activation"]
            for phrase in activation_phrases:
                ratio = fuzz.ratio(cmd, phrase)
                if ratio > max_ratio:
                    max_ratio = ratio
                    res_command = command

        if max_ratio >= self.min_command_confidence:
            command = commands[res_command]
            self.command_recognized_signal.emit(command["name"], cmd)
            if "reply" in command.keys():
                self.say(command["reply"])

    def say(self, text):
        audio = self.tts_model.apply_tts(ssml_text=f'<speak><prosody rate="slow">{text}</prosody></speak>', #text + "..",
                                speaker=self.speaker,
                                sample_rate=self.sample_rate,
                                put_accent=True,
                                put_yo=True)
        self.mute_mic = True # глушим микрофон
        sd.play(audio, self.sample_rate)
        time.sleep((len(audio) / self.sample_rate) + 0.5)
        sd.stop()
        del audio
        self.mute_mic = False  # отключаем глушилку микрофона
