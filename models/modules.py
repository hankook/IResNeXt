import torch
import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, *size):
        super(View, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(x.size()[0], *self.size)

class ChannelPadding(nn.Module):
    def __init__(self, num_channels):
        super(ChannelPadding, self).__init__()
        self.num_channels = num_channels

    def forward(self, x):
        return F.pad(x, (0, 0, 0, 0, 0, self.num_channels))

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        h, w = x.size()[2], x.size()[3]
        return F.avg_pool2d(x, (h, w))

def conv1x1(in_channels, out_channels, stride=1, group=1):
    return nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False, stride=stride, groups=group)

def conv3x3(in_channels, out_channels, stride=1, group=1):
    return nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False, stride=stride, groups=group)
