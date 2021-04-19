from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox, QFileDialog, \
    QTextBrowser, QLabel
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtGui
from PySide2.QtCore import Signal, QObject
from PIL import Image
import torch.utils.data
from torchvision.utils import save_image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from Module.GAN测试 import generator
import Module.测试 as RegconizeNum     #模块从Module包里导入“测试”
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #没gpu的话就用cpu

class MySignals(QObject):
    # 定义一种信号，参赛是str，即文件的地址
    loadlabel = Signal(str)

global_ms = MySignals()  # 实例化信号

#class InputNumWindow():

class ImgWindow():  # 显示图片的窗口
    def __init__(self):
        super().__init__()
        # 使用ui文件导入定义界面类
        self.ui = QUiLoader().load('img_window.ui')
        self.ui.Button_exit.clicked.connect(self.exit_b)  #
        global_ms.loadlabel.connect(self.load_img)  # 连接信号与槽

    def exit_b(self):
        os.remove("temp.png")  # 删除生成的临时文件
        self.ui.close()

    def load_img(self, object):
        print(object)
        im = Image.open(object)  # 这里把原来的jpg转化成png之后打开
        im.save('temp.png')
        pixmap = QtGui.QPixmap('temp.png')
        label = self.ui.img_label
        label.setPixmap(pixmap)  # 加载图片
        label.setScaledContents(True)  # 自适应


class MainWindow():  # 主窗口
    def __init__(self):
        super().__init__()
        # 使用ui文件导入定义界面类
        self.ui = QUiLoader().load('my_ui.ui')
        self.ui.Button_loadmodel.clicked.connect(self.load_model)
        self.ui.Button_openimg.clicked.connect(self.open_img)
        #self.ui.Button_randnum.clicked.connect(self.input_randnum)
        self.ui.Button_consequence.clicked.connect(self.predict_res)

    def load_model(self):
        FileDialog = QFileDialog(self.ui.Button_loadmodel)  # 实例化
        FileDialog.setFileMode(QFileDialog.AnyFile)  # 可以打开任何文件
        model_file, _ = FileDialog.getOpenFileName(self.ui.Button_loadmodel, 'open file', './',
                                                   'model files (*.pth)')
        # 改变Text里面的文字
        self.ui.View_model_log.setPlainText("成功加载模型\n模型路径:" + model_file)

    def open_img(self):  # 这里和load_model差不多
        FileDialog = QFileDialog(self.ui.Button_openimg)
        FileDialog.setFileMode(QFileDialog.AnyFile)
        image_file, _ = FileDialog.getOpenFileName(self.ui.Button_openimg, 'open file', './',
                                                   'Image files (*.jpg *.gif *.png *.jpeg)')
        if not image_file:
            QMessageBox.warning(self.ui.Button_openimg, "警告", "文件错误或打开文件失败！", QMessageBox.Yes)
            return
        self.ui.View_img_log.setPlainText("成功加载图片\n图片路径:" + image_file)
        self.window2 = ImgWindow()
        global_ms.loadlabel.emit(image_file)  # 注意只有先实例化之后 发送信号 对应的槽才会执行
        self.window2.ui.show()
        return image_file

    def input_randnum(self):
        DEVICE = "cuda"
        G = generator(100, 3136).to(device)
        my_net = torch.load(r"generator.pth",map_location=torch.device('cpu'))  # 加载模型,没gpu的话将内存定位cpu
        G.load_state_dict(my_net)
        num = 10  # 用户输入需要的图像张数
        for i in range(num):
            z = torch.randn(1, 100).to(device)
            # z = torch.ones(100).unsqueeze(0).to(device)
            # z = Variable(torch.randn(1, 100))
            fake_img = G(z)
            save_image(fake_img, './dccc_test/test_{0}.png'.format(i))

    def predict_res(self):
        image_file = self.ui.View_img_log.toPlainText().split('路径:')[1]
        img = cv2.imread(image_file, 0)  # 加载图像,RGB是3通道图像0表示以一通道的灰度图表示
        img = cv2.resize(img, (28, 28))  # 缩放到28*28
        img = RegconizeNum.nor(img)  # 归一化到0-1
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # 图像转为tensor格式
        model = RegconizeNum.ConvNet().to(DEVICE)
        model_file = self.ui.View_model_log.toPlainText().split('路径:')[1]  # 模型保存地址
        my_net = torch.load(model_file)
        model.load_state_dict(my_net)
        model.eval()
        output = model(img)  # 预测
        pred = output.max(1, keepdim=True)[1]
        self.ui.View_predict_log.setPlainText("预测识别的结果为：" + str(pred[0][0].data.cpu().numpy()))


app = QApplication([])
start = MainWindow()
start.ui.show()
app.exec_()
