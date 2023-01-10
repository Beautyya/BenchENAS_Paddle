# coding=utf-8
import paddle
from train.learningrate.learningrate import BaseLearningRate


class CosineAnnealingLR(BaseLearningRate):
    """CosineAnnealingLR
    """

    def __init__(self, **kwargs):
        super(CosineAnnealingLR, self).__init__(**kwargs)

    def get_learning_rate(self):
        return paddle.optimizer.lr.CosineAnnealingDecay(self.lr, int(self.current_epoch))
