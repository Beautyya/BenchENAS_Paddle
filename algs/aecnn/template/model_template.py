"""
from __future__ import print_function
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ResNetBottleneck(nn.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, self.expansion * planes, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetUnit(nn.Layer):
    def __init__(self, amount, in_channel, out_channel):
        super(ResNetUnit, self).__init__()
        self.in_planes = in_channel
        self.layer = self._make_layer(ResNetBottleneck, out_channel, amount, stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out


class DenseNetBottleneck(nn.Layer):
    def __init__(self, nChannels, growthRate):
        super(DenseNetBottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2D(nChannels)
        self.conv1 = nn.Conv2D(nChannels, interChannels, kernel_size=1,
                               bias_attr=False)
        self.bn2 = nn.BatchNorm2D(interChannels)
        self.conv2 = nn.Conv2D(interChannels, growthRate, kernel_size=3,
                               padding=1, bias_attr=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = paddle.concat(x=[x, out], axis=1)
        return out


class DenseNetUnit(nn.Layer):
    def __init__(self, k, amount, in_channel, out_channel, max_input_channel):
        super(DenseNetUnit, self).__init__()
        self.out_channel = out_channel
        if in_channel > max_input_channel:
            self.need_conv = True
            self.bn = nn.BatchNorm2D(in_channel)
            self.conv = nn.Conv2D(in_channel, max_input_channel, kernel_size=1, bias_attr=False)
            in_channel = max_input_channel

        self.layer = self._make_dense(in_channel, k, amount)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for _ in range(int(nDenseBlocks)):
            layers.append(DenseNetBottleneck(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        if hasattr(self, 'need_conv'):
            out = self.conv(F.relu(self.bn(out)))
        out = self.layer(out)
        assert (out.shape[1] == self.out_channel)
        return out


class EvoCNNModel(nn.Layer):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        # generated_init

    def forward(self, x):
        x = self.bn1(self.conv(x))
        # generate_forward

        out = paddle.reshape(out, [out.shape[0], -1])
        out = self.linear(out)
        return out
"""