import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from utils import plot_6image,one_hot,plot_curve
import torchvision


class Net(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super(Net, self).__init__()
        # 定义第一层，输入为28*28，输出为256
        self.fc1 = nn.Linear(28 * 28, 256)
        # 定义第二层，输入为256，输出为64
        self.fc2 = nn.Linear(256, 64)
        # 定义第三层，输入为64，输出为10
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 第一层，激活函数为relu
        x = F.relu(self.fc1(x))
        # 第二层，激活函数为relu
        x = F.relu(self.fc2(x))
        # 第三层，激活函数为relu
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # 从网上下载mnist数据集,转为tensor后进行归一化操作，mean=0.1307,std=0.3081
    train_data_set = torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                                transform=torchvision.transforms.Compose(
                                                    [torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    # 创建dataloader,batchsize为512，需要打乱
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=512, shuffle=True)
    # 测试集，同理，
    test_data_set = torchvision.datasets.MNIST('mnist_data', train=False, download=True,
                                               transform=torchvision.transforms.Compose(
                                                   [torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    # 测试集不需要打乱
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=512, shuffle=True)
    # 查看数据集一个数据的信息
    images, lables = next(iter(train_loader))
    # 每个images里面有batchsize个元素，每个图片有一个通道，每个通道有28*28个像素
    # 每个lables里面有batchsize个真实标签
    print(images.shape, lables.shape, images.min(), images.max())
    # torch.Size([512, 1, 28, 28]) torch.Size([512]) tensor(-0.4242) tensor(2.8215)
    # 查看前六张图片
    # plot_6image(images, lables, 'image sample')
    net = Net()
    # 定义优化器，学习率为0.01
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # loss_list用于保存每次迭代的loss
    loss_list = []
    # 训练3个epoch
    for epoch in range(3):
        for batch_idx, (images,lables) in enumerate(train_loader):
            # 将x的形状从[batch_size,1,28,28]变为[512,784],送入全连接层
            images = images.view(images.size(0), 28 * 28)
            # 将y的形状从[batch_size]变为[batch_size,10],送入全连接层
            lables_one_hot = one_hot(lables)
            #进行预测
            predict = net(images)
            #计算损失
            loss = F.mse_loss(predict, lables_one_hot)
            #梯度清零
            optimizer.zero_grad()
            #反向传播
            loss.backward()
            #更新参数
            optimizer.step()
            #保存loss
            loss_list.append(loss.item())
            #每10个batch打印一次loss
            if batch_idx % 10 == 0:
                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
plot_curve(loss_list)
#进行测试
total_correct = 0
for images,lables in test_loader:
    images = images.view(images.size(0), 28 * 28)
    predict = net(images)
    #取出每行最大值的下标，即为预测的数字
    predict = torch.argmax(predict, dim=1)
    #计算预测正确的个数
    correct = torch.eq(predict, lables).sum().float().item()
    total_correct += correct
#计算准确率
accuracy = total_correct / len(test_data_set)
print('test accuracy:', accuracy)