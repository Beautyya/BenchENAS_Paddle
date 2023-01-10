"""
from __future__ import print_function
import argparse
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np
import os
import datetime

c0 = 16

class SeparableConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2D(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias_attr=bias)
        self.pointwise = nn.Conv2D(in_channels, out_channels, 1, 1, 0, 1, 1, bias_attr=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class DepthwiseConv2d(nn.Layer):
    def __init__(self, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseConv2d, self).__init__()

        self.conv1 = nn.Conv2D(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias_attr=bias)

    def forward(self, x):
        x = self.conv1(x)
        return x


class motif_1_1(nn.Layer):
    def __init__(self, in_channel = c0):
        super(motif_1_1, self).__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2D(in_channel)
        self.conv1_1 = nn.Conv2D(in_channel,in_channel,1,1)

    def forward(self, input):
        return self.relu(self.bn(self.conv1_1(input)))

class motif_1_2(nn.Layer):
    def __init__(self, in_channels=c0):
        super(motif_1_2, self).__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2D(in_channels)
        self.depwise_conv_3_3 = DepthwiseConv2d(in_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

    def forward(self, input):
        return self.relu(self.bn(self.depwise_conv_3_3(input)))

class motif_1_3(nn.Layer):
    def __init__(self, in_channel=c0):
        super(motif_1_3, self).__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2D(in_channel)
        self.conv3_3 = SeparableConv2d(in_channel,in_channel,3,1,1)

    def forward(self, input):
        return self.relu(self.bn(self.conv3_3(input)))


class motif_1_4(nn.Layer):
    def __init__(self, in_channel=c0):
        super(motif_1_4, self).__init__()
        self.pool = nn.MaxPool2D(3, stride=1, padding=1)

    def forward(self, input):
        return self.pool(input)


class motif_1_5(nn.Layer):
    def __init__(self, in_channel=c0):
        super(motif_1_5, self).__init__()
        self.pool = nn.AvgPool2D(3, stride=1, padding=1)

    def forward(self, input):
        return self.pool(input)

class motif_1_6(nn.Layer):
    def __init__(self, in_channel=c0):
        super(motif_1_6, self).__init__()

    def forward(self, input):
        return input

# generated_motif


class CellModel(nn.Layer):
    def __init__(self, in_channel=c0):
        super(CellModel, self).__init__()
        # generated_init
        self.in_channel = in_channel
        self.layer1 = motif_3_1(in_channel)


    def forward(self, input):
        # generate_forward


class EvoCNNModel(nn.Layer):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        # generate_conv
        self.cellmodel1 = CellModel(16)
        self.cellmodel2 = CellModel(32)
        self.cellmodel3 = CellModel(64)
        self.sep_conv3x3_2_1 = SeparableConv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.sep_conv3x3_2_2 = SeparableConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.sep_conv3x3_1 = SeparableConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2D(3, stride=1, padding=1)
        self.dropout = nn.Dropout()
        # linear_layer


    def forward(self, input):
        x = self.conv3x3(input)
        x = self.cellmodel1(x)
        x = self.sep_conv3x3_2_1(x)
        x = self.cellmodel2(x)
        x = self.sep_conv3x3_2_2(x)
        x = self.cellmodel3(x)
        x = self.sep_conv3x3_1(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.dropout(x)
        x = self.last_linear(x)
        x = nn.Softmax(1)(x)
        return x
"""
