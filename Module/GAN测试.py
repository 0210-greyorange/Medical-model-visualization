
import torch.utils.data
from torchvision.utils import save_image
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # batch, 3136=1x56x56
        self.br = nn.Sequential(
            nn.ReLU(True),
            nn.InstanceNorm2d(1)
            #nn.BatchNorm2d(1),
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),  # batch, 50, 56, 56
            nn.ReLU(True),
            #nn.InstanceNorm2d(50)
            nn.BatchNorm2d(50)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 56, 56
            nn.ReLU(True),
            #nn.InstanceNorm2d(25)
            nn.BatchNorm2d(25),
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),  # batch, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x

if __name__ == '__main__':
    DEVICE="cuda"
    G=generator(100, 3136).to(device)
    my_net = torch.load("generator.pth", map_location=torch.device('cpu'))  # 加载模型
    G.load_state_dict(my_net)
    num=10   # 用户输入需要的图像张数
    for i in range(num):
        z = torch.randn(1, 100).to(device)
        #z = torch.ones(100).unsqueeze(0).to(device)
        #z = Variable(torch.randn(1, 100))
        fake_img = G(z)
        save_image(fake_img, './dccc_test/test_{0}.png'.format(i))