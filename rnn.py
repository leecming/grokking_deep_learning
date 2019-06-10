"""
Common RNN layers
"""

import numpy as np
from activations import Sigmoid, Tanh
from tensor import Tensor
from layers import Layer, Linear


class RNNCell(Layer):
    """
    Vanilla RNN implementation
    Hidden(t) = Activation(Linear(Hidden(t-1) + Linear(Input(t)))
    Output(t) = Linear(Hidden(t))
    """
    def __init__(self,
                 n_inputs,
                 n_hidden,
                 n_output,
                 activation='sigmoid'):
        """
        :param n_inputs: dimension of inputs
        :param n_hidden: dimension of hidden layer
        :param n_output: dimension of output (token)
        :param activation: either sigmoid or tanh
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        if activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'tanh':
            self.activation = Tanh()
        else:
            raise Exception("Non-linearity not found")

        self.w_ih = Linear(n_inputs, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)

        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()

    def forward(self, input_tensor, hidden):
        """ Forward prop - returns both the output and the hidden """
        from_prev_hidden = self.w_hh.forward(hidden)
        combined = self.w_ih.forward(input_tensor) + from_prev_hidden
        new_hidden = self.activation.forward(combined)
        output = self.w_ho.forward(new_hidden)
        return output, new_hidden

    def init_hidden(self, batch_size=1):
        """ First hidden state is all zeros"""
        return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)


class LSTMCell(Layer):
    """
    Base LSTM implementation:
    Cell(t) = forget_gate * Cell(t-1) + input_gate * update
    where forget_gate, input_gate are Linear(prev_hidden)+Linear(input) and sigmoided
    and update is Linear(prev_hidden)+Linear(input) and tanhed
    Output(t) = output_gate * Tanh(Cell(t))
    """
    def __init__(self, n_inputs, n_hidden, n_output):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.xf = Linear(n_inputs, n_hidden)
        self.xi = Linear(n_inputs, n_hidden)
        self.xo = Linear(n_inputs, n_hidden)
        self.xc = Linear(n_inputs, n_hidden)
        self.hf = Linear(n_hidden, n_hidden, bias=False)
        self.hi = Linear(n_hidden, n_hidden, bias=False)
        self.ho = Linear(n_hidden, n_hidden, bias=False)
        self.hc = Linear(n_hidden, n_hidden, bias=False)

        self.w_ho = Linear(n_hidden, n_output, bias=False)

        self.parameters += self.xf.get_parameters()
        self.parameters += self.xi.get_parameters()
        self.parameters += self.xo.get_parameters()
        self.parameters += self.xc.get_parameters()
        self.parameters += self.hf.get_parameters()
        self.parameters += self.hi.get_parameters()
        self.parameters += self.ho.get_parameters()
        self.parameters += self.hc.get_parameters()

        self.parameters += self.w_ho.get_parameters()

    def forward(self, current_input, hidden):
        """
        updates cell, and returns the current time step's output,
        new hidden state, and the updated cell
        """
        prev_hidden = hidden[0]
        prev_cell = hidden[1]

        f = (self.xf.forward(current_input) + self.hf.forward(prev_hidden)).sigmoid()
        i = (self.xi.forward(current_input) + self.hi.forward(prev_hidden)).sigmoid()
        o = (self.xo.forward(current_input) + self.ho.forward(prev_hidden)).sigmoid()
        g = (self.xc.forward(current_input) + self.hc.forward(prev_hidden)).tanh()
        cell = (f * prev_cell) + (i * g)
        h = o * cell.tanh()

        output = self.w_ho.forward(h)
        return output, (h, cell)

    def init_hidden(self, batch_size=1):
        """ inits both hidden and cell to all zeros """
        h = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        c = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        h.data[:, 0] += 1
        c.data[:, 0] += 1

        return h, c
