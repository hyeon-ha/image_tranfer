import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PIL import Image # pillow 설치
import numpy as np
from tensorflow.keras.models import load_model
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization

form_window = uic.loadUiType('./job07_gan_app2.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pixmaplist= ['./app_image/result_house_mone3.jpg', './app_image/result_house_sen7.jpg', './app_image/go_pearl.jpg', './app_image/monet_pearl.jpg',
                          './app_image/matis_pearl.jpg', './app_image/matis_pearl2.jpg']
        self.setWindowTitle('Gan지 미술관')
        self.output1.setPixmap(QPixmap('./app_image/result_house_mone3.jpg'))
        self.output1.hide()

        self.background.setPixmap(QPixmap('app_image/배경.jpg'))
        # 투명도 조절
        opacity = QGraphicsOpacityEffect(self.background)
        opacity.setOpacity(0.5)
        self.background.setGraphicsEffect(opacity)

        self.buttons_flag = [True, True, True, True, True, True]

        self.setWindowIcon(QIcon('./app_image/icon.png'))



        self.before.hide()
        self.next.hide()
        self.original.hide()
        self.style.hide()
        self.setStyleSheet('color:black; background:rgb(240,255,255)')
        self.image_muse.clicked.connect(self.image_gan)

        self.open_file.clicked.connect(self.open_file_slot)
        self.mone_style.clicked.connect(self.mone_style_slot)
        self.next.clicked.connect(self.next_button_slot)
        self.before.clicked.connect(self.before_button_slot)
        self.original.clicked.connect(self.original_transfer)
        self.style.clicked.connect(self.transfer_image)

    def image_gan(self):
        self.output1.setPixmap(QPixmap('./app_image/result_house_mone3.jpg'))
        # self.original.hide()
        self.next.show()
        self.before.show()
        self.output1.show()
        self.buttons_flag[0] = False
        self.original.hide()
        self.style.hide()

    def before_button_slot(self):
        for i in range(len(self.buttons_flag)):
            if self.buttons_flag[i] == False:
                self.output1.setPixmap(QPixmap(self.pixmaplist[i - 1]))
                self.buttons_flag[i] = True
                self.buttons_flag[i - 1] = False
                break

    def next_button_slot(self):
        for i in range(len(self.buttons_flag)-1):
            if self.buttons_flag[i] == False:
                self.output1.setPixmap(QPixmap(self.pixmaplist[i+1]))
                self.buttons_flag[i] = True
                self.buttons_flag[i + 1] = False
                break

    def open_file_slot(self):
        self.fname = QFileDialog.getOpenFileName(self)
        print(self.fname[0])
        pm = QPixmap(self.fname[0])
        self.output1.setPixmap(pm)
        self.output1.show()
        self.next.hide()
        self.before.hide()
        image = Image.open("{}".format(self.fname[0]))
        image.save("filename.jpeg")
        self.original.hide()
        self.style.hide()
        # pixmap_1.setPixmap(pixmap)
        # self.image_generator(fname[0])

    def mone_style_slot(self):
        try:
            img = Image.open(self.fname[0])
            img = img.convert('RGB')  # RGB모드로 변환
            img = img.resize((256, 256))  # 사이즈는 튜플로
            data = np.asarray(img) / 127.5 - 1  # 이미지를 어레이로 바꾼다.
            data = np.expand_dims(data, axis=0)  # np 확장시켜서 넣는다.
            print('loading...')
            G_AB = load_model('./keras_models/cycle_ganA_epoch1904.h5')
            print('trans....')
            fake_B=G_AB.predict(data)
            gen_imgs = np.concatenate([data, fake_B])
            print('working....')
            gen_imgs = 0.5 * gen_imgs + 0.5
            plt.imsave('filename2.jpeg', gen_imgs[1])
            pm = QPixmap('./filename2.jpeg')
            self.output1.setPixmap(pm)
            # self.original.show()
            self.original.show()
        except:
            pass

    def original_transfer(self):
        pm = QPixmap('./filename.jpeg')
        self.output1.setPixmap(pm)
        self.style.show()
        self.original.hide()

    def transfer_image(self):
        pm = QPixmap('./filename2.jpeg')
        self.output1.setPixmap(pm)
        self.original.show()
        self.style.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())