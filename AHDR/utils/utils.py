import numpy as np
import torch
import random


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_parameters(model):
    """
    初始化网络参数
    :param model:
    :return:
    """
    # 对网络内所有的层分别进行不同的初始化
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            # 对卷积层进行初始化,使用kaiming_normal_初始化
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                # 如果存在偏置项，对偏置项进行初始化,设置为0
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            # 对BN层进行初始化
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            # 对全连接层进行初始化
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0)


def set_random_seed(seed):
    """
    设置随机种子,为numpy和torch设置随机种子
    :param seed:
    :return:
    """
    # 设置numpy的随机种子
    random.seed(seed)
    np.random.seed(seed)
    # 设置pytorch的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
