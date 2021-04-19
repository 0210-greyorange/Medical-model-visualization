import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
def nor(img,min=0,max=1):
    image_new = (img - np.min(img)) * (max - min) / (np.max(img) - np.min(img)) + min
    return image_new
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5) # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = nn.Conv2d(10, 20, 3) # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20*10*10, 500) # 输入通道数是2000，输出通道数是500
        self.fc2 = nn.Linear(500, 10) # 输入通道数是500，输出通道数是10，即10分类
    def forward(self,x):
        in_size = x.size(0) # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
        out = self.conv1(x) # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = F.relu(out) # batch*10*24*24（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2) # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = self.conv2(out) # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out) # batch*20*10*10
        out = out.view(in_size, -1) # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = self.fc1(out) # batch*2000 -> batch*500
        out = F.relu(out) # batch*500
        out = self.fc2(out) # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1) # 计算log(softmax(x))
        return out


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    model=ConvNet().to(DEVICE)
    my_net = torch.load(r"number.pth") # 模型保存地址
    model.load_state_dict(my_net)
    model.eval()


    img=cv2.imread(r"6.png",0)   # 加载图像,RGB是3通道图像0表示以一通道的灰度图表示
    img=cv2.resize(img,(28,28))   # 缩放到28*28
    img=nor(img)               # 归一化到0-1
    print(img.shape)
    import matplotlib.pyplot as plt     #  展示这张图像
    plt.imshow(img,cmap="gray")
    plt.show()

    img=torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(DEVICE)    # 图像转为tensor格式

    output = model(img)    # 预测
    pred = output.max(1, keepdim=True)[1]
    print(pred[0][0].data.cpu().numpy())    # 预测结果为tensor格式，转为numpy数值形式输出，这个值为返回值