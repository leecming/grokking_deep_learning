import numpy as np
np.random.seed(0)


class Layer:
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters


class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim

        weight = (np.random.rand(vocab_size, dim) - 0.5) / dim
        self.weight = Tensor(weight, autograd=True)

        self.parameters.append(self.weight)

    def forward(self, input):
        return self.weight.index_select(input)


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


class CrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input.cross_entropy(target)


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
    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))

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

                if self.creation_op == 'index_select':
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad))

                if self.creation_op == 'cross_entropy':
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))

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

    def index_select(self, indices):
        if self.autograd:
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op='index_select')
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])

    def cross_entropy(self, target_indices):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape)-1,
                                       keepdims=True)
        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = - (np.log(p) * target_dist).sum(1).mean()

        if self.autograd:
            out = Tensor(loss,
                         autograd=True,
                         creators=[self],
                         creation_op='cross_entropy')
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out
        return Tensor(loss)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


class RNNCell(Layer):
    def __init__(self,
                 n_inputs,
                 n_hidden,
                 n_output,
                 activation='sigmoid'):
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

    def forward(self, input, hidden):
        from_prev_hidden = self.w_hh.forward(hidden)
        combined = self.w_ih.forward(input) + from_prev_hidden
        new_hidden = self.activation.forward(combined)
        output = self.w_ho.forward(new_hidden)
        return output, new_hidden

    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size,self.n_hidden)),autograd=True)


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

# x = Tensor(np.eye(5), autograd=True)
# x.index_select(Tensor([[1, 2, 3], [2, 3, 4]])).backward()
# print(x.grad)


# data = Tensor(np.array([1, 2, 1, 2]), autograd=True)
# target = Tensor(np.array(([0], [1], [0], [1])), autograd=True)
#
# embed = Embedding(5, 3)
# model = Sequential([embed, Tanh(), Linear(3, 1), Sigmoid()])
# criterion = MSELoss()
#
# optim = SGD(parameters=model.get_parameters(), alpha=0.5)
#
# for i in range(10):
#     pred = model.forward(data)
#     loss = criterion.forward(pred, target)
#     loss.backward(Tensor(np.ones_like(loss.data)))
#     optim.step()
#     print(loss)

# data = Tensor(np.array([1, 2, 1, 2]), autograd=True)
# target = Tensor(np.array([0, 1, 0, 1]), autograd=True)
#
# model = Sequential([Embedding(3, 3), Tanh(), Linear(3, 4)])
# criterion = CrossEntropyLoss()
# optim = SGD(parameters=model.get_parameters(), alpha=0.1)
#
# for i in range(10):
#     pred = model.forward(data)
#     loss = criterion.forward(pred, target)
#     loss.backward(Tensor(np.ones_like(loss.data)))
#     optim.step()
#     print(loss)

import sys,random,math
from collections import Counter

with open('../data/tasksv11/en/qa1_single-supporting-fact_train.txt', 'r') as f:
    raw = f.readlines()

tokens = list()
for line in raw[:1000]:
    tokens.append(line.lower().replace("\n", "").split(" ")[1:])

new_tokens = list()
for line in tokens:
    new_tokens.append(['-'] * (6 - len(line)) + line)
tokens = new_tokens

vocab = set()
for sent in tokens:
    for word in sent:
        vocab.add(word)
vocab = list(vocab)

word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i


def words2indices(sentence):
    idx = list()
    for word in sentence:
        idx.append(word2index[word])
    return idx


indices = list()
for line in tokens:
    idx = list()
    for w in line:
        idx.append(word2index[w])
    indices.append(idx)

data = np.array(indices)

embed = Embedding(vocab_size=len(vocab),dim=16)
model = RNNCell(n_inputs=16, n_hidden=16, n_output=len(vocab))
criterion = CrossEntropyLoss()
params = model.get_parameters() + embed.get_parameters()
optim = SGD(parameters=params, alpha=0.05)

for iter in range(1000):
    batch_size = 100
    total_loss = 0

    hidden = model.init_hidden(batch_size=batch_size)

    for t in range(5):
        input = Tensor(data[0:batch_size, t], autograd=True)
        rnn_input = embed.forward(input=input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)

    target = Tensor(data[0:batch_size, t+1], autograd=True)
    loss = criterion.forward(output, target)
    loss.backward()
    optim.step()
    total_loss += loss.data
    if iter % 200 == 0:
        p_correct = (target.data == np.argmax(output.data,axis=1)).mean()
        print_loss = total_loss / (len(data)/batch_size)
        print("Loss:", print_loss, "% Correct:", p_correct)
