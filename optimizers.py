"""
Common optimizer functions
"""


class SGD:
    """ Basic SGD with specified alpha """
    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        """ Zero all param tensor grads"""
        for curr_param in self.parameters:
            curr_param.grad.data *= 0

    def step(self, zero=True):
        """ runs through all parameters, updating their weights"""
        for curr_param in self.parameters:
            curr_param.data -= curr_param.grad.data * self.alpha

            # zero out the tensors' gradients after we're done with the update
            if zero:
                curr_param.grad.data *= 0
