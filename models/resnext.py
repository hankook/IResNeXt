import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.modules import View, ChannelPadding, GlobalAvgPool as GAP, conv1x1, conv3x3

__all__ = ['ResNeXt', 'ResNet']

class GrouppedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck, group):
        super(GrouppedBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck = bottleneck
        self.group = group

        bottleneck *= group
        self.conv1 = conv1x1(in_channels, bottleneck)
        self.conv2 = conv3x3(bottleneck, bottleneck, group=group)
        self.conv3 = conv1x1(bottleneck, out_channels)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(bottleneck)
        self.bn3 = nn.BatchNorm2d(bottleneck)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        x = self.conv2(self.relu(self.bn2(x)))
        x = self.conv3(self.relu(self.bn3(x)))

        return x

class BasicBlock(GrouppedBlock):
    def __init__(self, in_channels, out_channels, bottleneck):
        super(BasicBlock, self).__init__(in_channels, out_channels, bottleneck, 1)

class ResNeXt(nn.Module):
    def __init__(self, num_blocks=3, num_classes=10, width=64, bottleneck=4, cardinality=8, pretrained=None, num_copies=None):
        super(ResNeXt, self).__init__()
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.width = width
        self.bottleneck = bottleneck
        self.cardinality = cardinality

        w, b, c = width, bottleneck, cardinality
        self.transitions = nn.ModuleList([conv3x3(3, w)])
        for i in range(2):
            self.transitions.append(nn.Sequential(nn.AvgPool2d(2), ChannelPadding(w<<i)))
        self.stages = nn.ModuleList()
        for i in range(3):
            self.stages.append(nn.ModuleList())
            for j in range(num_blocks):
                self.stages[i].append(GrouppedBlock(w<<i, w<<i, b<<i, c))
        self.output_layer = nn.Sequential(nn.BatchNorm2d(w<<2),
                                          nn.ReLU(inplace=True),
                                          nn.AvgPool2d(8),
                                          View(-1),
                                          nn.Linear(w<<2, num_classes))

        if pretrained is not None:
            assert num_copies > 0 and c % num_copies == 0

            weights = np.concatenate([np.random.uniform(size=(3, num_blocks, num_copies-1)),
                                      np.ones((3, num_blocks, 1))], axis=2)
            weights = np.sort(weights)
            for i in range(3):
                for j in range(num_blocks):
                    for k in reversed(range(1, num_copies)):
                        weights[i, j, k] -= weights[i, j, k-1]
            self.weights = weights
            print(weights)

            c2 = c / num_copies
            own_state = self.state_dict()
            pretrained_state = torch.load(pretrained, map_location='cpu')
            for name, param in pretrained_state.items():
                if 'transitions' in name or 'output_layer' in name or 'bn1' in name:
                    own_state[name].copy_(param)
                else:
                    _, i, j, _, _ = name.split('.')
                    i, j = int(i), int(j)
                    u = c2*(b<<i)
                    if 'conv3' not in name:
                        for k in range(num_copies):
                            own_state[name][k*u:(k+1)*u].copy_(param)
                    else:
                        for k in range(num_copies):
                            own_state[name][:, k*u:(k+1)*u].copy_(param*weights[i, j, k])

    def forward(self, x):
        for i in range(3):
            x = self.transitions[i](x)
            for block in self.stages[i]:
                x = x+block(x)
        x = self.output_layer(x)
        return x

class ResNet(ResNeXt):
    def __init__(self, num_blocks=3, num_classes=10, width=64, bottleneck=16):
        super(ResNet, self).__init__(num_blocks, num_classes, width, bottleneck, 1)
