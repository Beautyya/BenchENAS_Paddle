# coding=utf-8

import paddle
from train.optimizer.optimizer import BaseOptimizer


class RMSprop(BaseOptimizer):
    """RMSprop optimizer
    """

    def __init__(self, **kwargs):
        super(RMSprop, self).__init__(**kwargs)

    def get_optimizer(self, weight_params):
        return paddle.optimizer.RMSProp(parameters=weight_params, learning_rate=self.lr)
