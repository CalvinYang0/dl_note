import torch
import torch.nn as nn
import my_models.ahdr as ahdr
import utils.utils as utils
import os
import my_models.loss as loss
import my_dataset.dataset_sig17 as dataset
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='HDR-Transformer',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_dir", type=str, default='./data',
                        help='dataset directory'),
    parser.add_argument('--patch_size', type=int, default=256),
    parser.add_argument("--sub_set", type=str, default='sig17_training_crop128_stride64',
                        help='dataset directory')
    parser.add_argument('--logdir', type=str, default='./checkpoints',
                        help='target log directory')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data (default: 8)')
    # Training
    parser.add_argument('--resume', type=str, default='./checkpoints/val_latest_checkpoint.pth',
                        help='load model from a .pth file')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=443, metavar='S',
                        help='random seed (default: 443)')
    parser.add_argument('--init_weights', action='store_true', default=False,
                        help='init model weights')
    parser.add_argument('--loss_func', type=int, default=1,
                        help='loss functions for training')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--lr_decay_interval', type=int, default=100,
                        help='decay learning rate every N epochs(default: 100)')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='start epoch of training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='training batch size (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser.parse_args()


def adjust_learning_rate(args, optimizer, epoch):
    # 将优化器内所有被优化参数的learningrate都进行衰减，
    lr = args.lr * (0.5 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    args = get_args()
    # 随机参数
    if args.seed is not None:
        utils.set_random_seed(args.seed)
    # 设置随机种子
    utils.set_random_seed(args.seed)
    # 检测当前目录下是否存在用于保存监测点的文件夹，如果没有则创建一个
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    # 选择训练设备
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    # 实例化训练网络
    model = ahdr.AHDRNet(channel_in=6, channel_train=64, densenet_num=6, growth_rate=64).to(device)

    # 当前psnr最高指标
    current_psnr = [-1.0]
    # 初始化训练参数
    if args.init_weights:
        utils.init_parameters(model)
    # 设置损失函数
    criterion = loss.L1MuLoss().to(device)
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    # 如果需要重新读取则从上一次保存的结果开始
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("===> Loading checkpoint from: {}".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("===> Loaded checkpoint: epoch {}".format(checkpoint['epoch']))
    #     else:
    #         print("===> No checkpoint is founded at {}.".format(args.resume))
    # 如果多卡则开启并行
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # 加载数据集
    train_dataset = dataset.SIG17_Training_Dataset(root_dir=args.dataset_dir, sub_set=args.sub_set, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_dataset = dataset.SIG17_Validation_Dataset(root_dir=args.dataset_dir, is_training=False, crop=True,
                                                   crop_size=512)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    # 训练集长度
    dataset_size = len(train_dataset)

    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch, criterion, )
        # 记录checkpoint
        test(args, model, device, optimizer, epoch, current_psnr)


def test_single_img(model, img_dataset, device):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, shuffle=False, num_workers=1)
    # 关闭梯度计算，节省内存
    with torch.no_grad():
        # len dataloader为单张图片分块后的长度，逐个取出后推测，拼成完整的pred
        for batch_data in tqdm(dataloader, total=len(dataloader)):
            batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), \
                batch_data['input2'].to(device)
            output = model(batch_ldr0, batch_ldr1, batch_ldr2)
            # 将输出结果保存到img_dataset中,并将其转换为numpy格式,detach()将其从计算图中分离出来，去掉batch_size为1的信息
            img_dataset.update_result(torch.squeeze(output.detach().cpu()).numpy().astype(np.float32))
    pred, label = img_dataset.rebuild_result()
    return pred, label


def train(args, model, device, train_loader, optimizer, epoch, criterion):
    # 开启训练模式
    model.train()
    # 训练时间均值
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    # 结束时间，初始化为当前时间
    end = time.time()
    # 长度为训练集长度的进度条
    with tqdm(total=train_loader.__len__()) as pbar:
        for batch_index, batch_data in enumerate(train_loader):
            # 更新数据开始时间
            data_time.update(time.time() - end)
            # 获取数据
            ldr0, ldr1, ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), batch_data[
                'input2'].to(device)
            # 读取标签
            label = batch_data['label'].to(device)
            # 读取预测值
            pred = model(ldr0, ldr1, ldr2)
            # 计算损失
            loss = criterion(pred, label)
            # 梯度清空
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 优化器更新参数
            optimizer.step()
            # 更新训练时间，一个batch完成计算的时间
            batch_time.update(time.time() - end)
            # 当前batch结束，更新结束时间
            end = time.time()
            # 间隔200个batch打印一次训练信息
            if batch_index % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:3f})\t'
                      'Data: {data_time.val:.3f} ({data_time.avg:3f})'.format(
                    epoch,
                    batch_index * args.batch_size,
                    len(train_loader.dataset),
                    100. * batch_index * args.batch_size / len(train_loader.dataset),
                    loss.item(),
                    batch_time=batch_time,
                    data_time=data_time
                ))
            # 更新进度条,并打印当前loss和epoch
            pbar.set_postfix(loss=loss.item(), epoch=epoch)
            # 更新进度条,1表示进度条前进1个单位 s1
            pbar.update(1)


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """
    calculate SSIM

    :param img1: [0, 255]
    :param img2: [0, 255]
    :return:
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def test(args, model, device, optimizer, epoch, cur_psnr):
    # 开启验证模式
    model.eval()
    # 读取测试集
    test_datasets = dataset.SIG17_Test_Dateset(args.dataset_dir, args.patch_size)
    # 记录psnr，ssim
    psnr_l = utils.AverageMeter()
    ssim_l = utils.AverageMeter()
    psnr_mu = utils.AverageMeter()
    ssim_mu = utils.AverageMeter()
    for idx, img_dataset in enumerate(test_datasets):
        # 返回的img_dataset包含一个数据信息
        pred_img, label = test_single_img(model, img_dataset, device)
        scene_psnr_l = peak_signal_noise_ratio(label, pred_img, data_range=1.0)

        label_mu = loss.np_range_compressor(label)
        pred_img_mu = loss.np_range_compressor(pred_img)

        scene_psnr_mu = peak_signal_noise_ratio(label_mu, pred_img_mu, data_range=1.0)
        # 将三通道放到最后rgb
        pred_img = np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0)
        label = np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0)
        pred_img_mu = np.clip(pred_img_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        label_mu = np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0)

        scene_ssim_l = calculate_ssim(pred_img, label)  # H W C data_range=0-255
        scene_ssim_mu = calculate_ssim(pred_img_mu, label_mu)
        psnr_l.update(scene_psnr_l)
        ssim_l.update(scene_ssim_l)
        psnr_mu.update(scene_psnr_mu)
        ssim_mu.update(scene_ssim_mu)
        print('==Validation==\tPSNR_l: {:.4f}\t PSNR_mu: {:.4f}\t SSIM_l: {:.4f}\t SSIM_mu: {:.4f}'.format(
            psnr_l.avg,
            psnr_mu.avg,
            ssim_l.avg,
            ssim_mu.avg
        ))

    # 保存的信息
    save_dict = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    # 将这一次的结果保存
    torch.save(save_dict, os.path.join("./checkpoints", 'val_latest_checkpoint.pth'))
    # 如果模型更优则保存
    if psnr_mu.avg > cur_psnr[0]:
        torch.save(save_dict, os.path.join("./checkpoints", 'best_checkpoint.pth'))
        cur_psnr[0] = psnr_mu.avg
        # 将参数写入json文件
        with open(os.path.join("./checkpoints", 'best_checkpoint.json'), 'w') as f:
            f.write('best epoch:' + str(epoch) + '\n')
            f.write('Validation set: Average PSNR: {:.4f}, PSNR_mu: {:.4f}, SSIM_l: {:.4f}, SSIM_mu: {:.4f}\n'.format(
                psnr_l.avg,
                psnr_mu.avg,
                ssim_l.avg,
                ssim_mu.avg
            ))


if __name__ == '__main__':
    main()
