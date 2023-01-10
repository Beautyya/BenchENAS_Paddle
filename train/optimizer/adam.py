# coding=utf-8

import paddle
from train.optimizer.optimizer import BaseOptimizer


class Adam(BaseOptimizer):
    """Adam optimizer
    """
    def __init__(self, **kwargs):
        super(Adam, self).__init__(**kwargs)

    def get_optimizer(self, weight_params):
        return paddle.optimizer.Adam(parameters=weight_params, learning_rate=self.lr)

