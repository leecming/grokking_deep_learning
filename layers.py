"""
Convenience classes that compose common Tensor operations
into layers
1. Embedding - creates an embedding matrix and does index lookups during forward prop
2. Linear - Dense layer with weight and optional bias
3. Sequential - composes multiple layers
"""

import numpy as np
from tensor import Tensor


class Layer:
    """
    Base class specifying all layers need to track and show their params
    """
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        """
        Tracks tensor parameters for weights updates
        :return:
        """
        return self.parameters


class Embedding(Layer):
    """
    Embedding matrix with lookup operation
    Simple weight init, weighted by number of dimensions
    """
    def __init__(self, vocab_size, dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim

        weight = (np.random.rand(vocab_size, dim) - 0.5) / dim
        self.weight = Tensor(weight, autograd=True)

        self.parameters.append(self.weight)

    def forward(self, indices):
        """ forward op is lookup of embed matrix"""
        return self.weight.index_select(indices)


class Linear(Layer):
    """
    Dense layer with optional bias weights
    Simple weights init weighted by input dimensions
    """
    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()

        self.use_bias = bias

        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.weight = Tensor(W, autograd=True)

        if self.use_bias:
            self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)

        if self.use_bias:
            self.parameters.append(self.bias)

    def forward(self, input_matrix):
        """ Input dot Weights + bias"""
        if self.use_bias:
            return input_matrix.dot(self.weight)+self.bias.expand(0,
                                                                  len(input_matrix.data))
        return input_matrix.dot(self.weight)


class Sequential(Layer):
    """
    Model-level wrapper for layers a la Keras
    """
    def __init__(self, layers=None):
        super().__init__()
        self.layers = list() if layers is None else layers

    def add(self, layer):
        """ Add specified layer to model """
        self.layers.append(layer)

    def forward(self, input_matrix):
        """
        Calls its layers' forward in sequence
        """
        for layer in self.layers:
            input_matrix = layer.forward(input_matrix)
        return input_matrix

    def get_parameters(self):
        """ Aggregate all its layers' params"""
        params = list()
        for curr_layer in self.layers:
            params += curr_layer.get_parameters()
        return params
