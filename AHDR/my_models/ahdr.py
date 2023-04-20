import torch
import torch.nn as nn
import torch.nn.functional as F


class make_dilation_dense(nn.Module):
    """
    扩张残差模块
    """

    def __init__(self, channel_in, growth_rate, kernel_size):
        super(make_dilation_dense, self).__init__()
        # 扩张后的卷积核，保持每通道信息大小不变
        self.conv = nn.Conv2d(channel_in, growth_rate, kernel_size=kernel_size, stride=1,
                              padding=(kernel_size - 1) // 2 + 1, dilation=2)

    def forward(self, x):
        out = self.conv(x)
        # densenet，将输入和输出拼接，会导致增加通道数，每次增加growth_grate
        out = torch.cat((x, out), 1)
        return out


class DRDB(nn.Module):
    """
    DRDB模块
    """

    def __init__(self, channel_in, densenet_num, growth_rate):
        super(DRDB, self).__init__()
        modules = []
        present_channel = channel_in
        # 按照要求的densenet的数量引入densenet
        for i in range(densenet_num):
            modules.append(make_dilation_dense(present_channel, growth_rate, 3))
            # 每增加一个模块总通道数就会上升
            present_channel += growth_rate
        # 将所有的densenet模块组合成一个序列
        self.dense_layers = nn.Sequential(*modules)
        # 将通道数降回到channel_in
        self.conv_1x1 = nn.Conv2d(present_channel, channel_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 将输入的x传入densenet模块
        out = self.dense_layers(x)
        # 将输出的通道数降回到channel_in
        out = self.conv_1x1(out)
        # 将输入的x和降维后的x相加
        out = x + out
        return out


class AHDRNet(nn.Module):
    def __init__(self, channel_in, channel_train, densenet_num, growth_rate):
        """n
        AHDRNet网络模型
        :param channel_in:输入的图形通道数
        :param channel_train: 中间训练的通道数
        :param densenet_num为 用于训练的DBRB模块的数量
        :param growth_rate: DBRB 中densenet的增长率
        """
        super(AHDRNet, self).__init__()
        # 将输入的图像通道数channel_in，输出的通道数channel_train，卷积核大小为3，步长为1，填充为1
        # 通过这个卷积核从三张3通道的ldr图像获得64通道的z1，z2，z3，z2为参考图像写作zr
        self.seq_init = nn.Sequential(nn.Conv2d(channel_in, channel_train, kernel_size=3, stride=1, padding=1),
                                      nn.LeakyReLU())
        # 对zi，zr concat后的结果进行卷积，relu，卷积，sigmoid，或则attention模块
        self.seq_make_att1 = nn.Sequential(
            nn.Conv2d(channel_train * 2, channel_train * 2, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
            nn.Conv2d(channel_train * 2, channel_train, kernel_size=3, stride=1, padding=1), nn.Sigmoid())
        self.seq_make_att2 = nn.Sequential(
            nn.Conv2d(channel_train * 2, channel_train * 2, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
            nn.Conv2d(channel_train * 2, channel_train, kernel_size=3, stride=1, padding=1), nn.Sigmoid())
        # 用于将attention后的结果转成drdb模块输入
        self.conv_init_drdb = nn.Conv2d(channel_train * 3, channel_train, kernel_size=3, stride=1, padding=1)
        # 用于训练的DBRB模块
        self.drdb1 = DRDB(channel_train, densenet_num, growth_rate)
        self.drdb2 = DRDB(channel_train, densenet_num, growth_rate)
        self.drdb3 = DRDB(channel_train, densenet_num, growth_rate)
        # 用于3次drdb后的结果降维
        self.GFF_1x1 = nn.Conv2d(channel_train * 3, channel_train, kernel_size=1, stride=1, padding=0)
        self.GFF_3x3 = nn.Conv2d(channel_train, channel_train, kernel_size=3, stride=1, padding=1)
        # 将z2融合后的结果和drdb模块后的结果累加后进行卷积
        self.conv_up = nn.Conv2d(channel_train, channel_train, kernel_size=3, stride=1, padding=1)
        # 将最后的结果进行卷积，得到最终的输出
        self.conv_out = nn.Conv2d(channel_train, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, non_reference1, reference2, non_reference3):
        # 获取初始输入
        z1 = self.seq_init(non_reference1)
        z2 = self.seq_init(reference2)
        z3 = self.seq_init(non_reference3)
        # attention模块attention1，concat，卷积，relu，卷积，sigmoid
        attention1 = self.seq_make_att1(torch.cat((z1, z2), dim=1))
        # attention模块a3，concat，卷积，relu，卷积，sigmoid
        attention3 = self.seq_make_att2(torch.cat((z3, z2), dim=1))
        # 通过attention1和attention3对z1和z3进行加权
        z1 = z1 * attention1
        z3 = z3 * attention3
        # 将z1，z2，z3进行concat
        zs = torch.cat((z1, z2, z3), dim=1)
        # 通过DBRB模块进行训练
        f0 = self.conv_init_drdb(zs)
        f1 = self.drdb1(f0)
        f2 = self.drdb2(f1)
        f3 = self.drdb3(f2)
        # 将训练后的结果进行concat
        f = torch.cat((f1, f2, f3), dim=1)
        # 通过GFF模块进行降维
        FdLF = self.GFF_1x1(f)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + z2
        us = self.conv_up(FDF)
        # 将训练后的结果进行卷积，得到最终的输出
        out = self.conv_out(us)
        output = torch.sigmoid(out)
        return output
