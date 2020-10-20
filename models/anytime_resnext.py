import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.modules import View, ChannelPadding, conv1x1, conv3x3
from utils.measure_v2 import add_flops, add_params

__all__ = ['AnytimeResNeXt']

class AnytimeGrouppedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck, group, anytime):
        super(AnytimeGrouppedBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck = bottleneck
        self.group = group
        self.anytime = anytime

        assert group % anytime == 0

        bottleneck *= group
        self.conv1 = conv1x1(in_channels, bottleneck)
        self.conv2 = conv3x3(bottleneck, bottleneck, group=group)
        # self.conv3 = conv1x1(bottleneck, out_channels*anytime, group=anytime)
        self.conv3 = conv1x1(bottleneck, out_channels)

        self.bn1 = nn.ModuleList()
        for i in range(anytime):
            self.bn1.append(nn.BatchNorm2d(in_channels))
        self.bn2 = nn.ModuleList()
        for i in range(anytime):
            self.bn2.append(nn.BatchNorm2d(bottleneck*(i+1)//anytime))
        self.bn3 = nn.ModuleList()
        for i in range(anytime):
            self.bn3.append(nn.BatchNorm2d(bottleneck*(i+1)//anytime))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, k):
        assert k < self.anytime

        b = self.bottleneck * self.group
        h, w = x.size(2), x.size(3)

        x = self.relu(self.bn1[k](x))
        x = F.conv2d(x, self.conv1.weight[:b*(k+1)//self.anytime])

        x = self.relu(self.bn2[k](x))
        x = F.conv2d(x, self.conv2.weight[:b*(k+1)//self.anytime], padding=1, groups=self.group//self.anytime*(k+1))

        x = self.relu(self.bn3[k](x))
        x = F.pad(x, (0, 0, 0, 0, 0, b-x.size()[1]))
        x = self.conv3(x)
        add_flops((b*(k+1)//self.anytime-b)*x.numel())
        add_params((b*(k+1)//self.anytime-b)*x.size(1))
        # x = F.conv2d(x, self.conv3.weight[:self.out_channels*(k+1)], groups=k+1)
        # x = x.view(-1, k+1, self.out_channels, h, w)
        # x = x.sum(1)

        return x

class AnytimeResNeXt(nn.Module):
    def __init__(self, num_blocks=3, num_classes=10, width=64, bottleneck=4, cardinality=8, anytime=8, init=False):
        super(AnytimeResNeXt, self).__init__()
        self.num_blocks = num_blocks
        self.anytime = anytime
        self.cardinality = cardinality
        self.width = width
        self.num_classes = num_classes
        self.bottleneck = bottleneck

        w, b, c = width, bottleneck, cardinality
        self.transitions = nn.ModuleList([conv3x3(3, w)])
        for i in range(2):
            self.transitions.append(nn.Sequential(nn.AvgPool2d(2), ChannelPadding(w<<i)))

        self.stages = nn.ModuleList()
        for i in range(3):
            self.stages.append(nn.ModuleList())
            for j in range(num_blocks):
                self.stages[i].append(AnytimeGrouppedBlock(w<<i, w<<i, b<<i, c, anytime))

        self.output_layers = nn.ModuleList()
        for i in range(anytime):
            layer = nn.Sequential(nn.BatchNorm2d(w<<2),
                                  nn.ReLU(inplace=True),
                                  nn.AvgPool2d(8),
                                  View(-1),
                                  nn.Linear(w<<2, num_classes))
            self.output_layers.append(layer)

        if init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()

    def forward(self, x, k=-1):
        if k < 0: k += self.anytime
        y = x
        for i in range(3):
            y = self.transitions[i](y)
            for block in self.stages[i]:
                add_flops(y.numel())
                y = y+block(y, k)
        y = self.output_layers[k](y)
        return y
