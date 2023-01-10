# coding=utf-8
import paddle

from train.learningrate.learningrate import BaseLearningRate



class StepLR(BaseLearningRate):
    """StepLR
    """

    def __init__(self, **kwargs):
        super(StepLR, self).__init__(**kwargs)

    def get_learning_rate(self):
        return paddle.optimizer.lr.StepDecay(self.lr, step_size=10, gamma=0.1)

