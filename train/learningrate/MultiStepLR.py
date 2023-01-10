# coding=utf-8
import paddle

from train.learningrate.learningrate import BaseLearningRate



class MultiStepLR(BaseLearningRate):
    """MultiStepLR
    """

    def __init__(self, **kwargs):
        super(MultiStepLR, self).__init__(**kwargs)

    def get_learning_rate(self):
        return paddle.optimizer.lr.MultiStepDecay(self.lr, milestones=[10, 15, 25, 30], gamma=0.1)

