"""
Common activations
"""
from layers import Layer


class Tanh(Layer):
    """Tanh"""
    @staticmethod
    def forward(input_tensor):
        """ actual computation within Tensor class """
        return input_tensor.tanh()


class Sigmoid(Layer):
    """Sigmoid"""
    @ staticmethod
    def forward(input_tensor):
        """ actual computation within Tensor class """
        return input_tensor.sigmoid()
