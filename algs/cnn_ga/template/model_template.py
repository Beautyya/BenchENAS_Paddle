"""
from __future__ import print_function
import paddle
import paddle.nn as nn
from paddle.vision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor
import paddle.nn.functional as F
import os
from datetime import datetime
import multiprocessing

class BasicBlock(nn.Layer):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class EvoCNNModel(nn.Layer):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        # generate_init


    def forward(self, x):
        # generate_forward

        out = paddle.reshape(out, [out.shape[0], -1])
        out = self.linear(out)
        return out

"""


