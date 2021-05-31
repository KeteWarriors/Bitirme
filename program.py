import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDir
from facialemotionrecognition import *
from facereg import *
sonuc = [None] * 11


class Ui_MainWindow(QWidget):
    foto_name = ""
    fotolar = []
    photowidgets = []
    textwidgets = []

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1149, 701)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(650, 540, 113, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.topluFotografSec)

        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(790, 540, 113, 32))
        self.pushButton_3.setObjectName("pushButton_3")

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(930, 540, 113, 32))
        self.pushButton_4.setObjectName("pushButton_4")

        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(420, 30, 741, 491))
        self.groupBox.setObjectName("groupBox")

        self.photo_11 = QtWidgets.QLabel(self.groupBox)
        self.photo_11.setGeometry(QtCore.QRect(360, 330, 151, 131))
        self.photo_11.setText("")
        self.photo_11.setPixmap(QtGui.QPixmap("code.png"))
        self.photo_11.setScaledContents(True)
        self.photo_11.setObjectName("photo_11")

        self.photo_5 = QtWidgets.QLabel(self.groupBox)
        self.photo_5.setGeometry(QtCore.QRect(520, 30, 151, 131))
        self.photo_5.setText("")
        self.photo_5.setPixmap(QtGui.QPixmap("code.png"))
        self.photo_5.setScaledContents(True)
        self.photo_5.setObjectName("photo_5")

        self.photo_10 = QtWidgets.QLabel(self.groupBox)
        self.photo_10.setGeometry(QtCore.QRect(200, 330, 151, 131))
        self.photo_10.setText("")
        self.photo_10.setPixmap(QtGui.QPixmap("code.png"))
        self.photo_10.setScaledContents(True)
        self.photo_10.setObjectName("photo_10")

        self.photo_8 = QtWidgets.QLabel(self.groupBox)
        self.photo_8.setGeometry(QtCore.QRect(360, 180, 151, 131))
        self.photo_8.setText("")
        self.photo_8.setPixmap(QtGui.QPixmap("code.png"))
        self.photo_8.setScaledContents(True)
        self.photo_8.setObjectName("photo_8")

        self.photo_6 = QtWidgets.QLabel(self.groupBox)
        self.photo_6.setGeometry(QtCore.QRect(40, 180, 151, 131))
        self.photo_6.setText("")
        self.photo_6.setPixmap(QtGui.QPixmap("code.png"))
        self.photo_6.setScaledContents(True)
        self.photo_6.setObjectName("photo_6")

        self.photo_4 = QtWidgets.QLabel(self.groupBox)
        self.photo_4.setGeometry(QtCore.QRect(360, 30, 151, 131))
        self.photo_4.setText("")
        self.photo_4.setPixmap(QtGui.QPixmap("code.png"))
        self.photo_4.setScaledContents(True)
        self.photo_4.setObjectName("photo_4")

        self.photo_2 = QtWidgets.QLabel(self.groupBox)
        self.photo_2.setGeometry(QtCore.QRect(40, 30, 151, 131))
        self.photo_2.setText("")
        self.photo_2.setPixmap(QtGui.QPixmap("code.png"))
        self.photo_2.setScaledContents(True)
        self.photo_2.setObjectName("photo_2")

        self.photo_3 = QtWidgets.QLabel(self.groupBox)
        self.photo_3.setGeometry(QtCore.QRect(200, 30, 151, 131))
        self.photo_3.setText("")
        self.photo_3.setPixmap(QtGui.QPixmap("code.png"))
        self.photo_3.setScaledContents(True)
        self.photo_3.setObjectName("photo_3")

        self.photo_9 = QtWidgets.QLabel(self.groupBox)
        self.photo_9.setGeometry(QtCore.QRect(520, 180, 151, 131))
        self.photo_9.setText("")
        self.photo_9.setPixmap(QtGui.QPixmap("code.png"))
        self.photo_9.setScaledContents(True)
        self.photo_9.setObjectName("photo_9")

        self.photo_7 = QtWidgets.QLabel(self.groupBox)
        self.photo_7.setGeometry(QtCore.QRect(200, 180, 151, 131))
        self.photo_7.setText("")
        self.photo_7.setPixmap(QtGui.QPixmap("code.png"))
        self.photo_7.setScaledContents(True)
        self.photo_7.setObjectName("photo_7")

        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(90, 160, 60, 16))
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(250, 160, 60, 16))
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(410, 160, 60, 16))
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(570, 160, 60, 16))
        self.label_4.setObjectName("label_4")

        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(90, 310, 60, 16))
        self.label_5.setObjectName("label_5")

        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(250, 310, 60, 16))
        self.label_6.setObjectName("label_6")

        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(400, 310, 60, 16))
        self.label_7.setObjectName("label_7")

        self.label_8 = QtWidgets.QLabel(self.groupBox)
        self.label_8.setGeometry(QtCore.QRect(560, 310, 60, 16))
        self.label_8.setObjectName("label_8")

        self.label_9 = QtWidgets.QLabel(self.groupBox)
        self.label_9.setGeometry(QtCore.QRect(250, 460, 60, 16))
        self.label_9.setObjectName("label_9")

        self.label_10 = QtWidgets.QLabel(self.groupBox)
        self.label_10.setGeometry(QtCore.QRect(410, 460, 60, 16))
        self.label_10.setObjectName("label_10")

        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(40, 30, 371, 581))
        self.groupBox_2.setObjectName("groupBox_2")

        self.photo = QtWidgets.QLabel(self.groupBox_2)
        self.photo.setGeometry(QtCore.QRect(10, 30, 351, 441))
        self.photo.setText("")
        self.photo.setPixmap(QtGui.QPixmap("code.png"))
        self.photo.setScaledContents(True)
        self.photo.setObjectName("photo")

        self.pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton.setGeometry(QtCore.QRect(110, 520, 113, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.fotografSec)

        self.label_11 = QtWidgets.QLabel(self.groupBox_2)
        self.label_11.setGeometry(QtCore.QRect(140, 480, 60, 16))
        self.label_11.setObjectName("label_11")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1149, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_2.setText(_translate("MainWindow", "Fotograf Seç"))
        self.pushButton_3.setText(_translate(
            "MainWindow", "Duygu Karşılaştır"))
        self.pushButton_4.setText(_translate("MainWindow", "Yüz Karşılaştır"))
        self.pushButton_3.clicked.connect(self.duygukarsilastir)
        self.pushButton_4.clicked.connect(self.yuzkarsilastir)
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.label.setText(_translate("MainWindow", sonuc[0]))
        self.label_2.setText(_translate("MainWindow", sonuc[1]))
        self.label_3.setText(_translate("MainWindow", sonuc[2]))
        self.label_4.setText(_translate("MainWindow", sonuc[3]))
        self.label_5.setText(_translate("MainWindow", sonuc[4]))
        self.label_6.setText(_translate("MainWindow", sonuc[5]))
        self.label_7.setText(_translate("MainWindow", sonuc[6]))
        self.label_8.setText(_translate("MainWindow", sonuc[7]))
        self.label_9.setText(_translate("MainWindow", sonuc[8]))
        self.label_10.setText(_translate("MainWindow", sonuc[9]))
        self.groupBox_2.setTitle(_translate("MainWindow", "GroupBox"))
        self.pushButton.setText(_translate("MainWindow", "Fotograf Seç"))
        self.label_11.setText(_translate("MainWindow", sonuc[10]))
        self.textwidgets = [self.label,
                            self.label_2,
                            self.label_3,
                            self.label_4,
                            self.label_5,
                            self.label_6,
                            self.label_7,
                            self.label_8,
                            self.label_9,
                            self.label_10,
                            self.label_11]

    def fotografSec(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Open Image File', r"/Users/OFY/Desktop/", "image files (*.jpg *.png *.jpeg)")
        self.photo.setPixmap(QPixmap(file_name))
        self.foto_name = file_name

    def topluFotografSec(self):
        self.photowidgets = fotograflar = [self.photo_2, self.photo_3, self.photo_4, self.photo_5, self.photo_6,
                                           self.photo_7, self.photo_8, self.photo_9, self.photo_10, self.photo_11]
        for i in range(len(fotograflar)):
            file_name, _ = QFileDialog.getOpenFileName(
                self, 'Open Image File', r"/Users/OFY/Desktop/", "image files (*.jpg *.png *.jpeg)")
            fotograflar[i].setPixmap(QPixmap(file_name))
            self.fotolar.append(file_name)

    def duygukarsilastir(self):

        for item in range(10):
            sonuc[item] = (karsilastir(self.fotolar[item]))
        sonuc[10] = karsilastir(self.foto_name)
        for item in range(11):
            self.textwidgets[item].setText(sonuc[item])
        for item in range(10):
            if sonuc[10] == sonuc[item]:
                self.photowidgets[item].setStyleSheet(
                    "border: 4px solid green;")
            else:
                self.photowidgets[item].setStyleSheet("border: 4px solid red;")

    def yuzkarsilastir(self):

        for item in range(len(self.fotolar)):
            sonuc[item] = (displayImages(self.foto_name, self.fotolar[item]))

        for item in range(10):
            self.textwidgets[item].setText(sonuc[item])
        for item in range(10):
            if "Benzer Kişiler" == sonuc[item]:
                self.photowidgets[item].setStyleSheet(
                    "border: 4px solid green;")
            else:
                self.photowidgets[item].setStyleSheet("border: 4px solid red;")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

