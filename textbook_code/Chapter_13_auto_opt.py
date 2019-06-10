import numpy as np
np.random.seed(0)


class Layer:
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.weight = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)

        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.mm(self.weight)+self.bias.expand(0, len(input.data))


class Sequential(Layer):
    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params


class MSELoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return ((pred - target) * (pred - target)).sum(0)


class SGD:
    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha

    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0

    def step(self, zero=True):
        for p in self.parameters:
            p.data -= p.grad.data * self.alpha

            if zero:
                p.grad.data *= 0


class Tensor:
    def __init__(self,
                 data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators  # list of parent Tensor objects
        self.grad = None
        self.autograd = autograd
        self.children = {}  # mapping of child id : count of gradients received
        if id is None:
            id = np.random.randint(0, 100000)
        self.id = id

        # add itself to creators' children mappings
        if creators is not None:
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    # confirm node has received correct number of gradients from each child
    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if cnt != 0:
                return False
        return True

    # grad_origin is the Tensor object of the child flowing gradient back to it
    # might be None if we're arbitrary dumping a constant gradient to a node
    def backward(self, grad, grad_origin=None):
        if self.autograd:
            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    raise Exception('cannot backprop more than once')
                else:
                    self.children[grad_origin.id] -= 1

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            # only flow gradients back to parents if it has received ALL gradients from its children
            if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None):
                if self.creation_op == 'add':
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)

                if self.creation_op == 'neg':
                    self.creators[0].backward(self.grad.__neg__())

                if self.creation_op == 'sub':
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad.__neg__(), self)

                if self.creation_op == 'mul':
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new, self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)

                if "sum" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.expand(dim,
                                                               self.creators[0].data.shape[dim]))

                if 'expand' in self.creation_op:
                    dim = int(self.creation_op.split('_')[1])
                    self.creators[0].backward(self.grad.sum(dim))

                if self.creation_op == 'transpose':
                    self.creators[0].backward(self.grad.transpose())

                if self.creation_op == 'mm':
                    act = self.creators[0]
                    weights = self.creators[1]
                    new = self.grad.mm(weights.transpose())
                    act.backward(new)
                    new = self.grad.transpose().mm(act).transpose()
                    weights.backward(new)

                if self.creation_op == 'sigmoid':
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (ones - self)))

                if self.creation_op == 'tanh':
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (ones - (self * self)))

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='add')
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='sub')
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='mul')
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op='sum_{}'.format(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)

        if self.autograd:
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op='expand_{}'.format(dim))
        return Tensor(new_data)

    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op='transpose')
        return Tensor(self.data.transpose())

    def mm(self, x):
        if self.autograd:
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op='mm')
        return Tensor(self.data.dot(x.data))

    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op='neg')
        return Tensor(self.data * -1)

    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op='sigmoid')
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op='tanh')
        return Tensor(np.tanh(self.data))

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


class Tanh(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sigmoid()

# a = Tensor([1, 2, 3, 4, 5], autograd=True)
# b = Tensor([2, 2, 2, 2, 2], autograd=True)
# c = Tensor([5, 4, 3, 2, 1], autograd=True)

# d = a + b
# e = b + c
# f = d + e
# f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
# print(b.grad)

# d = a + (-b)
# e = (-b) + c
# f = d + e
# f.backward(Tensor(np.array([1, 1, 1, 1, 1])))
# print(b.grad)

# data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
# target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)
#
# w = list()
# w.append(Tensor(np.random.rand(2, 3), autograd=True))
# w.append(Tensor(np.random.rand(3, 1), autograd=True))
#
# optim = SGD(parameters=w, alpha=0.1)
#
# for i in range(10):
#     pred = data.mm(w[0]).mm(w[1])
#     loss = ((pred - target) * (pred - target)).sum(0)
#     loss.backward(Tensor(np.ones_like(loss.data)))
#
#     optim.step()
#
#     print(loss)


data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

model = Sequential([Linear(2, 3), Tanh(), Linear(3, 1), Sigmoid()])
# model = Sequential([Linear(2, 3), Linear(3, 1)])
criterion = MSELoss()

optim = SGD(parameters=model.get_parameters(), alpha=1)

for i in range(10):
    pred = model.forward(data)
    loss = criterion.forward(pred, target)
    # loss = ((pred - target) * (pred - target)).sum(0)

    loss.backward(Tensor(np.ones_like(loss.data)))
    optim.step()
    print(loss)