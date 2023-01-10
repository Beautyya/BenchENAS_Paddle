# coding=utf-8
import paddle

from train.learningrate.learningrate import BaseLearningRate


class ExponentialLR(BaseLearningRate):
    """ExponentialLR
    """

    def __init__(self, **kwargs):
        super(ExponentialLR, self).__init__(**kwargs)

    def get_learning_rate(self):
        return paddle.optimizer.lr.ExponentialDecay(self.lr, gamma=0.2)
