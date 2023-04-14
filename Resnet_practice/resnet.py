import torch
from torch import nn
from torch.nn import functional as F


class ResnetBlock(nn.Module):
    """
    Resnet 模块
    """

    def __init__(self, channel_in, channel_out, stride):
        """

        :param channel_in:输入的通道数
        :param channel_out: 输出的通道数
        :param stride: 移动步长
        :return:
        """
        super(ResnetBlock, self).__init__()
        # 卷积层1，输入通道数为channel_in，输出通道数为channel_out，卷积核大小为3，步长为stride，填充为1
        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=stride, padding=1)
        # 批归一化层1
        self.bn1 = nn.BatchNorm2d(channel_out)
        # 卷积层2，输入通道数为channel_out，输出通道数为channel_out，卷积核大小为3，步长为1，填充为1
        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)
        # 批归一化层2
        self.bn2 = nn.BatchNorm2d(channel_out)
        # 额外的层，用于处理输入和输出通道数不一致的情况
        # 设置一个默认的空的序列
        self.extra = nn.Sequential()
        if channel_out != channel_in:
            # 如果输入通道数和输出通道数不一致，那么需要对输入进行处理
            # 将输入的通道数channel_in，输出的通道数channel_out，卷积核大小为1，步长为stride
            self.extra = nn.Sequential(
                nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channel_out)
            )

    def forward(self, x):
        """
        前向传播
        :param self:
        :param x:
        :return:
        """
        # 卷积层1
        out = F.relu(self.bn1(self.conv1(x)))
        # 卷积层2
        out = self.bn2(self.conv2(out))
        # 额外的层
        out = self.extra(x) + out
        # 激活函数
        out = F.relu(out)
        return out


class Resnet18(nn.Module):
    """
    Resnet18
    """

    def __init__(self):
        super(Resnet18, self).__init__()
        # 卷积层1，输入通道数为3，输出通道数为64，卷积核大小为3，步长为3，填充为0
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # resnet模块，分别为64，128，256，512
        self.blk1 = ResnetBlock(64, 128, stride=2)
        self.blk2 = ResnetBlock(128, 256, stride=2)
        self.blk3 = ResnetBlock(256, 512, stride=2)
        self.blk4 = ResnetBlock(512, 512, stride=2)
        # 全连接层
        self.fc = nn.Linear(512 * 2 * 2, 10)

    def forward(self, x):
        """
        前向传播
        :param self:
        :param x:
        :return:
        """
        # 卷积层1
        out = F.relu(self.conv1(x))
        # resnet模块
        out = self.blk1(out)
        out = self.blk2(out)
        out = self.blk3(out)
        out = self.blk4(out)
        # 输出层
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
