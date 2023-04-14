import torch
from matplotlib import pyplot as plt
import matplotlib
#应对plt无法输出的bug
matplotlib.use('TkAgg')
def plot_6image(img, lable, name):
    """
    将前六张图片显示出来
    :param img:
    :param lable:
    :param name:
    :return:
    """
    for i in range(6):
        # 现实的图像2行3列，第i+1个
        plt.subplot(2, 3, i + 1)
        # tight_layout()自动调整子图参数，使之填充整个图像区域
        plt.tight_layout()
        # cmap='gray'表示灰度图，由于在读入图像时已经进行了归一化，所以这里要反归一化显示原来的信息
        # interpolation='none'表示不进行插值，即不进行像素插值，显示原始的像素信息
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        # 显示图片的真实标签
        plt.title("{}: {}".format(name, lable[i].item()))
        # xticks()和yticks()分别表示x轴和y轴的刻度，这里不显示刻度
        plt.xticks([])
        plt.yticks([])
    # 显示图像
    plt.show()
def one_hot(lables,length=10):
    """
    将标签转换为one-hot编码
    :param lables:
    :param length:
    :return:
    """
    #输出形式为[batch_size,length]
    out=torch.zeros(lables.size(0),length)
    #将lables转为longTensor类型并转为列向量
    idx=torch.LongTensor(lables).view(-1,1)
    #将out中的对应位置的值设置为1
    out.scatter_(dim=1,index=idx,value=1)
    return out
def plot_curve(data):
    """
    绘制损失函数图像
    :param data:
    :return:
    """
    #创建一个图像
    fig=plt.figure()\
    #绘制图像,range(len(data))表示x轴的值，data表示y轴的值，color='blue'表示颜色为蓝色
    plt.plot(range(len(data)),data,color='blue')
    #显示图例，loc='upper right'表示图例显示在右上角
    plt.legend(['value'],loc='upper right')
    #x轴的标签
    plt.xlabel('step')
    #y轴的标签
    plt.ylabel('value')
    plt.show()

#%%
