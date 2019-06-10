"""
Common loss functions
"""
from layers import Layer


class MSELoss(Layer):
    """ mean squared error """
    @staticmethod
    def forward(pred, target):
        """ vanilla calculation """
        return ((pred - target) * (pred - target)).sum(0)


class CrossEntropyLoss(Layer):
    """ cross entropy error - already built into Tensor"""
    @staticmethod
    def forward(input_tensor, target_tensor):
        """ nothing sexy """
        return input_tensor.cross_entropy(target_tensor)
