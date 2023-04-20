import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
def np_range_compressor(hdr_img, mu=5000):
    # log(1+mu*hdr_img)/log(1+mu)
    return (np.log(hdr_img * mu + 1)) / np.log(mu + 1)
def range_compressor(hdr_img, mu=5000):
    # log(1+mu*hdr_img)/log(1+mu)
    return (torch.log(hdr_img * mu + 1)) / math.log(mu + 1)


class L1MuLoss(nn.Module):
    def __init__(self, mu=5000):
        super(L1MuLoss, self).__init__()
        self.mu = mu

    def forward(self, pred, lable):
        mu_pred = range_compressor(pred, self.mu)
        mu_lable = range_compressor(lable, self.mu)
        return nn.L1Loss()(mu_pred, mu_lable)
