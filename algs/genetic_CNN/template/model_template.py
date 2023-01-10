"""
from __future__ import print_function
import argparse
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
from paddle.vision import datasets, transforms
from paddle.optimizer.lr import StepDecay
import numpy as np
import os
import datetime

class ConvBlock(nn.Layer):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1):
        super(ConvBlock,self).__init__()
        self.conv_1 = nn.Conv2D(ch_in,ch_out,kernel_size,stride=stride,padding=1)
        self.bn_1 = nn.BatchNorm2D(ch_out)

    def  forward(self,x):
        out = F.relu(self.bn_1(self.conv_1(x)))
        return out



class EvoCNNModel(nn.Layer):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        self.pool = nn.MaxPool2D(3, stride=2, padding=1)
        self.relu = nn.ReLU()
        # generated_init


    def forward(self, input):
        # generate_forward

        out = paddle.flatten(input, 1)
        out = self.linear(out)
        output = F.log_softmax(out, axis=1)
        return output
"""
