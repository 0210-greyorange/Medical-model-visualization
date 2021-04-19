from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox, QFileDialog, \
    QTextBrowser, QLabel
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtGui
from PySide2.QtCore import Signal, QObject,QCoreApplication
from PIL import Image
import torch.utils.data
from torchvision.utils import save_image
import os
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from Module.随机生成数字 import generator
import Module.识别数字 as RegconizeNum  # 模块从Module包里导入“识别数字”

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 没gpu的话就用cpu


class MySignals(QObject):
    # 定义一种信号，参赛是str，即文件的地址
    ms = Signal(str)

global_ms = MySignals()  # 实例化信号
input_num_ms = MySignals()


class ImgWindow():  # 显示图片的窗口
    def __init__(self):
        super().__init__()
        # 使用ui文件导入定义界面类
        self.ui = QUiLoader().load('img_window.ui')
        self.ui.Button_exit.clicked.connect(self.exit_b)  #
        global_ms.ms.connect(self.load_img)  # 连接信号与槽

    def exit_b(self):
        os.remove("temp.png")  # 删除生成的临时文件
        self.ui.close()

    def load_img(self, object):
        im = Image.open(object)  # 这里把原来的jpg转化成png之后打开
        im.save('temp.png')
        pixmap = QtGui.QPixmap('temp.png')
        label = self.ui.img_label
        label.setPixmap(pixmap)  # 加载图片
        label.setScaledContents(True)  # 自适应


class InputNumWindow():  # 用户输入图片张数的窗口
    def __init__(self):
        self.ui = QUiLoader().load('input_num.ui')
        self.ui.ok_btn.clicked.connect(self.get_num)
        self.ui.cancel_btn.clicked.connect(self.close_ui)
    def get_num(self):
        num = self.ui.user_input_num.text()
        self.close_ui()
        input_num_ms.ms.emit(num)
        self.close_ui()
    def close_ui(self):
        self.ui.close()



class MainWindow():  # 主窗口
    def __init__(self):
        super().__init__()
        # 使用ui文件导入定义界面类
        self.ui = QUiLoader().load('my_ui.ui')
        self.ui.Button_loadmodel.clicked.connect(self.load_model)
        self.ui.Button_openimg.clicked.connect(self.open_img)
        self.ui.Button_randnum.clicked.connect(self.input_randnum)
        self.ui.Button_consequence.clicked.connect(self.predict_res)
        input_num_ms.ms.connect(self.input_randnum)

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
        image_file, _ = FileDialog.getOpenFileName(self.ui.Button_openimg, 'open file', './Handwriting num pic',
                                                   'Image files (*.jpg *.gif *.png *.jpeg)')
        if not image_file:
            QMessageBox.warning(self.ui.Button_openimg, "警告", "文件错误或打开文件失败！", QMessageBox.Yes)
            return
        self.ui.View_img_log.setPlainText("成功加载图片\n图片路径:" + image_file)
        self.window2 = ImgWindow()
        global_ms.ms.emit(image_file)  # 注意只有先实例化之后 发送信号 对应的槽才会执行
        self.window2.ui.show()

    def input_randnum(self,num_ms):
        self.window3 = InputNumWindow()
        self.window3.ui.show()
        num = int(num_ms)
        G = generator(100, 3136).to(device)
        model_file = self.ui.View_model_log.toPlainText().split('路径:')[1]  # 模型保存地址
        my_net = torch.load(model_file, map_location=torch.device('cpu'))  # 加载模型,没gpu的话将内存定位cpu
        G.load_state_dict(my_net)
        # num = 10 # 用户输入需要的图像张数
        filename = "Handwriting num pic"
        current_path = os.getcwd()  # 返回当前
        path_item = os.listdir(current_path)  # 返回（列表）将当前目录的所有内容
        picfile_path = "{}\Handwriting num pic".format(current_path)  # 图片保存进哪个文件夹的路径
        if filename not in path_item:
            os.mkdir(filename)  # 在当前目录创建文件夹

        for i in range(num):
            z = torch.randn(1, 100).to(device)
            # z = torch.ones(100).unsqueeze(0).to(device)
            # z = Variable(torch.randn(1, 100))
            fake_img = G(z)
            path = "./{}/pic_{}.png".format(filename, i + 1)  # 保存图片吗的路径
            save_image(fake_img, path.format(i))
            str = "成功生成{num}张手写数字图\n图片路径:{path}".format(num=num, path=picfile_path)
            self.ui.View_randnum_log.setPlainText(str)

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
