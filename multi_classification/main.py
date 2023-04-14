import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def forward(x):
    x = x @ w1.t() + b1
    x = F.relu(x)
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    return x


if __name__ == "__main__":
    batch_size = 200
    learning_rate = 0.01
    epochs = 10
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    # 初始化三层网络的参数
    w1, b1 = torch.randn(200, 784, requires_grad=True), \
        torch.zeros(200, requires_grad=True)
    w2, b2 = torch.randn(200, 200, requires_grad=True), \
        torch.zeros(200, requires_grad=True)
    w3, b3 = torch.randn(10, 200, requires_grad=True), \
        torch.zeros(10, requires_grad=True)
    # 初始化参数
    torch.nn.init.kaiming_normal_(w1)
    torch.nn.init.kaiming_normal_(w2)
    torch.nn.init.kaiming_normal_(w3)
    # 定义优化器
    optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 训练
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将28*28的图片展开成784的向量
            data = data.view(-1, 28 * 28)
            # 前向传播
            logits = forward(data)
            # 计算损失
            loss = criterion(logits, target)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
        print('epoch: {}, loss: {}'.format(epoch, loss.item()))
    # 测试总损失
    test_loss = 0
    # 测试正确数
    correct = 0
    # 测试
    for data, target in test_loader:
        # 将28*28的图片展开成784的向量
        data = data.view(-1, 28 * 28)
        # 前向传播
        logits = forward(data)
        # 计算损失
        test_loss += criterion(logits, target).item()
        # 计算预测值,返回最大值的索引就是返回这是数字几
        pred = logits.argmax(dim=1)
        # 计算正确数
        correct += pred.eq(target).sum()
    # 计算平均损失
    test_loss /= len(test_loader.dataset)
    # 计算准确率
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))


