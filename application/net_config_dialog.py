from PyQt6.QtWidgets import QDialog
from netifaces import interfaces
import platform

from net_config_dialog_ui import Ui_NetConfigDialog
import netifaces as ni

class NetConfigDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_NetConfigDialog()
        self.ui.setupUi(self)

        _interfaces = ni.interfaces()
        for _interface in _interfaces:
            try:
                _addr = ni.ifaddresses(_interface)[ni.AF_INET][0]['addr']
                self.ui.cbAddress.addItem(f"{_addr}")
            except:
                continue

    def get_host(self):
        return self.ui.cbAddress.currentText()

    def set_host(self, address):
        return self.ui.cbAddress.setCurrentText(address)

    def get_data_port(self):
        return self.ui.leDataPort.text()

    def set_data_port(self, port):
        return self.ui.leDataPort.setText(str(port))

    def get_video_port(self):
        return self.ui.leVideoPort.text()

    def set_video_port(self, port):
        return self.ui.leVideoPort.setText(str(port))
