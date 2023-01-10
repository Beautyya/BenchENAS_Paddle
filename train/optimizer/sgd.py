# coding=utf-8

import paddle
from train.optimizer.optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    """SGD optimizer
    """
    def __init__(self, **kwargs):
        super(SGD, self).__init__(**kwargs)

    def get_optimizer(self, weight_params):
        return paddle.optimizer.SGD(parameters=weight_params, learning_rate=self.lr)

