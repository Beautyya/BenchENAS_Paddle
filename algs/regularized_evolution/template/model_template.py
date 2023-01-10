"""

import paddle
import paddle.nn as nn
import collections

Genotype = collections.namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2D(3, stride=stride, padding=1),
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
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


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


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        ten = paddle.to_tensor(paddle.zeros([x.shape[0], 1, 1, 1]), dtype='float32')
        mask = paddle.to_tensor(paddle.bernoulli(ten), stop_gradient=False)
        x.divide(paddle.to_tensor(keep_prob, stop_gradient=False))
        x.multiply(mask)
    return x


class ChannelSELayer(nn.Layer):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias_attr=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias_attr=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.shape
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = paddle.multiply(input_tensor, paddle.reshape(fc_out_2, [a, b, 1, 1]))
        return output_tensor


class Cell(nn.Layer):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev, SE_layer=False):
        super(Cell, self).__init__()
        # print(C_prev_prev, C_prev, C)
        self._SE_layer_flag = SE_layer

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

        # because final is the concat of four intermediate nodes
        if SE_layer:
            self.SE_layer = ChannelSELayer(C * 4)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.LayerList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops.append(op)
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        out = paddle.concat(x=[states[i] for i in self._concat], axis=1)

        # SE layer
        if self._SE_layer_flag:
            out = self.SE_layer(out)
        return out


class AuxiliaryHeadCIFAR(nn.Layer):

    def __init__(self, C, num_classes):
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2D(5, stride=2, padding=0),  # image size = 2 x 2
            nn.Conv2D(C, 128, 1, bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 768, 2, bias_attr=False),
            nn.BatchNorm2D(768),
            nn.ReLU()
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(paddle.reshape(x,[x.shape[0], -1]))
        return x


class AuxiliaryHeadImageNet(nn.Layer):

    def __init__(self, C, num_classes):
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2D(5, stride=2, padding=0),
            nn.Conv2D(C, 128, 1, bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 768, 2, bias_attr=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU()
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(paddle.reshape(x,[x.shape[0], -1]))
        return x


class NetworkCIFAR(nn.Layer):

    def __init__(self, C_in, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2D(C_in, C_curr, 3, padding=1, bias_attr=False),
            nn.BatchNorm2D(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.LayerList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2D(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(paddle.reshape(out,[out.shape[0], -1]))
        return logits


class NetworkImageNet(nn.Layer):
    def __init__(self, C_in, C, num_classes, layers, auxiliary, genotype, SE_layer=False):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2D(C_in, C // 2, kernel_size=3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(C // 2),
            nn.ReLU(),
            nn.Conv2D(C // 2, C, 3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(C, C, 3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.LayerList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, SE_layer)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2D(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(paddle.reshape(out,[out.shape[0], -1]))
        return logits


class EvoCNNModel(nn.Layer):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        # geno

        self.net.drop_path_prob = 0.1

    def forward(self, x):
        return self.net(x)

"""