import  torch
from    torch.utils.data import DataLoader
from    torchvision import datasets
from    torchvision import transforms
from    torch import nn, optim
from  resnet import Resnet18
def main():
    batch_size = 128
    #将cifar数据集下载到cifar文件夹下，设置为训练集，设置数据预处理为，将图片大小调整为32*32，转换为tensor，归一化
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda')
    model = Resnet18().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1000):
        #开启训练模式
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):

            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, 'loss:', loss.item())
        #开启测试模式
        model.eval()
        #no_grad关闭梯度计算节省内存
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:

                x, label = x.to(device), label.to(device)

                logits = model(x)
                pred = logits.argmax(dim=1)
                #item()将tensor转换为python数据类型
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)
                # print(correct)

            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)
if __name__ == '__main__':
    main()