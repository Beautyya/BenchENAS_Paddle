"""
from __future__ import print_function
import paddle
from paddle import nn
import paddle.nn.functional as F
import math


def pair(value):
    res = list()
    for i in range(2):
        res.append(value)
    res = tuple(res)
    return res


class SamePad2d(nn.Layer):
    # Mimics tensorflow's 'SAME' padding.

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, inputs):
        in_width = inputs.shape[2]
        in_height = inputs.shape[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(inputs, [pad_left, pad_right, pad_top, pad_bottom], 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__

class EvoCNNModel(nn.Layer):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        # generated_init


    def forward(self, x):
        # generate_forward


if __name__ == '__main__':
    model = EvoCNNModel()
    # generate_summary

"""
