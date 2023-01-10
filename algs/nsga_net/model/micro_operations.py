import paddle
import paddle.nn as nn

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2D(3, stride=stride, padding=1, exclusive=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2D(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(),
        nn.Conv2D(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias_attr=False),
        nn.Conv2D(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias_attr=False),
        nn.BatchNorm2D(C, weight_attr=paddle.ParamAttr(trainable=affine), bias_attr=paddle.ParamAttr(trainable=affine))
    ),
}


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        ten = paddle.to_tensor(paddle.zeros([x.shape[0], 1, 1, 1]), dtype='float32')
        mask = paddle.to_tensor(paddle.bernoulli(ten), stop_gradient=False)
        x.divide(paddle.to_tensor(keep_prob, stop_gradient=False))
        x.multiply(mask)
    return x


class ReLUConvBN(nn.Layer):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(C_in, C_out, kernel_size, stride=stride, padding=padding, bias_attr=False),
            nn.BatchNorm2D(C_out, weight_attr=paddle.ParamAttr(trainable=affine), bias_attr=paddle.ParamAttr(trainable=affine))
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Layer):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias_attr=False),
            nn.Conv2D(C_in, C_out, kernel_size=1, padding=0, bias_attr=False),
            nn.BatchNorm2D(C_out, weight_attr=paddle.ParamAttr(trainable=affine), bias_attr=paddle.ParamAttr(trainable=affine)),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Layer):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias_attr=False),
            nn.Conv2D(C_in, C_in, kernel_size=1, padding=0, bias_attr=False),
            nn.BatchNorm2D(C_in, weight_attr=paddle.ParamAttr(trainable=affine), bias_attr=paddle.ParamAttr(trainable=affine)),
            nn.ReLU(),
            nn.Conv2D(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias_attr=False),
            nn.Conv2D(C_in, C_out, kernel_size=1, padding=0, bias_attr=False),
            nn.BatchNorm2D(C_out, weight_attr=paddle.ParamAttr(trainable=affine), bias_attr=paddle.ParamAttr(trainable=affine)),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Layer):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Layer):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].multiply(paddle.to_tensor(0., stop_gradient=False))


class FactorizedReduce(nn.Layer):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2D(C_in, C_out // 2, 1, stride=2, padding=0, bias_attr=False)
        self.conv_2 = nn.Conv2D(C_in, C_out // 2, 1, stride=2, padding=0, bias_attr=False)
        self.bn = nn.BatchNorm2D(C_out, weight_attr=paddle.ParamAttr(trainable=affine), bias_attr=paddle.ParamAttr(trainable=affine))

    def forward(self, x):
        x = self.relu(x)
        out = paddle.concat(x=[self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], axis=1)
        out = self.bn(out)
        return out


class SELayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias_attr=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias_attr=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = paddle.reshape(self.avg_pool(x), [b, c])
        y = paddle.reshape(self.fc(y), [b, c, 1, 1])
        return x * y.expand_as(x)
