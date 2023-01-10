"""
from __future__ import print_function
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optim
import os,argparse
import numpy as np

class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        #ANCHOR-generated_init


    def forward(self, x):
        #ANCHOR-generate_forward

        out = paddle.reshape(out, [out.shape[0], -1])
        out = self.linear(out)
        return out 

"""