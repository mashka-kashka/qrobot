# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(759, 627)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../icons/robot.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMinimumSize(QtCore.QSize(0, 320))
        self.tabWidget.setObjectName("tabWidget")
        self.tabCamera = QtWidgets.QWidget()
        self.tabCamera.setObjectName("tabCamera")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tabCamera)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gv_camera = QtWidgets.QGraphicsView(parent=self.tabCamera)
        self.gv_camera.setObjectName("gv_camera")
        self.gridLayout_3.addWidget(self.gv_camera, 0, 0, 1, 1)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../icons/eye.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.tabWidget.addTab(self.tabCamera, icon1, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        self.teLog = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.teLog.setFrameShape(QtWidgets.QFrame.Shape.Panel)
        self.teLog.setReadOnly(True)
        self.teLog.setObjectName("teLog")
        self.gridLayout.addWidget(self.teLog, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(parent=MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolBar)
        self.actionActivateRobot = QtGui.QAction(parent=MainWindow)
        self.actionActivateRobot.setCheckable(True)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../icons/robot-green.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        icon2.addPixmap(QtGui.QPixmap("../icons/robot-red.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On)
        self.actionActivateRobot.setIcon(icon2)
        self.actionActivateRobot.setObjectName("actionActivateRobot")
        self.actionConfig = QtGui.QAction(parent=MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../icons/gear.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionConfig.setIcon(icon3)
        self.actionConfig.setObjectName("actionConfig")
        self.actionExit = QtGui.QAction(parent=MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("../icons/quit.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        self.actionExit.setIcon(icon4)
        self.actionExit.setObjectName("actionExit")
        self.actionActivateComputer = QtGui.QAction(parent=MainWindow)
        self.actionActivateComputer.setCheckable(True)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("../icons/chip-green.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        icon5.addPixmap(QtGui.QPixmap("../icons/chip-red.png"), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On)
        self.actionActivateComputer.setIcon(icon5)
        self.actionActivateComputer.setObjectName("actionActivateComputer")
        self.toolBar.addAction(self.actionConfig)
        self.toolBar.addAction(self.actionActivateRobot)
        self.toolBar.addAction(self.actionActivateComputer)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionExit)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.actionExit.triggered.connect(MainWindow.close) # type: ignore
        self.actionActivateRobot.toggled['bool'].connect(MainWindow.on_activate_robot) # type: ignore
        self.actionConfig.triggered.connect(MainWindow.on_config) # type: ignore
        self.actionActivateComputer.toggled['bool'].connect(MainWindow.on_activate_computer) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Робот"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabCamera), _translate("MainWindow", "Камера"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionActivateRobot.setText(_translate("MainWindow", "Активировать робота"))
        self.actionActivateRobot.setIconText(_translate("MainWindow", "Активация"))
        self.actionActivateRobot.setStatusTip(_translate("MainWindow", "Активировать робота"))
        self.actionConfig.setText(_translate("MainWindow", "Настройки"))
        self.actionConfig.setStatusTip(_translate("MainWindow", "Открыть окно настроек робота"))
        self.actionExit.setText(_translate("MainWindow", "Выход"))
        self.actionExit.setStatusTip(_translate("MainWindow", "Завершить работу"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+X"))
        self.actionActivateComputer.setText(_translate("MainWindow", "Активировать компьютер"))
