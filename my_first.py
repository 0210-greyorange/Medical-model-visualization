from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QMessageBox, QFileDialog, \
    QTextBrowser, QLabel
from PySide2.QtUiTools import QUiLoader
from PySide2 import QtGui
from PySide2.QtCore import Signal, QObject
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


def nor(img, min=0, max=1):
    image_new = (img - np.min(img)) * (max - min) / (np.max(img) - np.min(img)) + min
    return image_new


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5)  # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = nn.Conv2d(10, 20, 3)  # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20 * 10 * 10, 500)  # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(500, 10)  # 输入通道数是500，输出通道数是10，即10分类

    def forward(self, x):
        in_size = x.size(0)  # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
        out = self.conv1(x)  # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = F.relu(out)  # batch*10*24*24（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2)  # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = self.conv2(out)  # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out)  # batch*20*10*10
        out = out.view(in_size, -1)  # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = self.fc1(out)  # batch*2000 -> batch*500
        out = F.relu(out)  # batch*500
        out = self.fc2(out)  # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1)  # 计算log(softmax(x))
        return out


class MySignals(QObject):
    # 定义一种信号，参赛是str，即文件的地址
    loadlabel = Signal(str)


global_ms = MySignals()  # 实例化信号


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

    def predict_res(self):
        image_file = self.ui.View_img_log.toPlainText().split('路径:')[1]
        img = cv2.imread(image_file, 0)  # 加载图像,RGB是3通道图像0表示以一通道的灰度图表示
        img = cv2.resize(img, (28, 28))  # 缩放到28*28
        img = nor(img)  # 归一化到0-1
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # 图像转为tensor格式
        model = ConvNet().to(DEVICE)
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
