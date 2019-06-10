import sys
import numpy as np
import uuid

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
        self.weight = Tensor(weight, autograd=True, name='embedding')

        self.parameters.append(self.weight)

    def forward(self, input):
        return self.weight.index_select(input)


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs, bias=True, name=None):
        super().__init__()

        self.use_bias = bias

        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.weight = Tensor(W, autograd=True, name=name)

        if self.use_bias:
            self.bias = Tensor(np.zeros(n_outputs), autograd=True, name=name)

        self.parameters.append(self.weight)

        if self.use_bias:
            self.parameters.append(self.bias)

    def forward(self, input):
        if self.use_bias:
            return input.mm(self.weight)+self.bias.expand(0, len(input.data))
        return input.mm(self.weight)


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
                 id=None,
                 name=None):
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators  # list of parent Tensor objects
        self.grad = None
        self.autograd = autograd
        self.children = {}  # mapping of child id : count of gradients received
        if id is None:
            id = np.random.randint(0, 10000000)
        self.id = id

        if name is None:
            self.name = '_'

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
                    # return
                    print('origin id: {}'.format(grad_origin.id))
                    print('self id: {}'.format(self.id))
                    print('self creation_op: {}'.format(self.creation_op))
                    print('# creators: {}'.format(len(self.creators)))
                    for c in self.creators:
                        print('Creator id {}, creation_op {}'.format(c.id, c.creation_op))
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
                    # self.creators[0].backward(self.grad.__neg__())
                    self.creators[0].backward(self.grad.__neg__(), self)

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
                    # self.creators[0].backward(self.grad.expand(dim, self.creators[0].data.shape[dim]))
                    self.creators[0].backward(self.grad.expand(dim, self.creators[0].data.shape[dim]), self)

                if 'expand' in self.creation_op:
                    dim = int(self.creation_op.split('_')[1])
                    # self.creators[0].backward(self.grad.sum(dim))
                    self.creators[0].backward(self.grad.sum(dim), self)

                if self.creation_op == 'transpose':
                    # self.creators[0].backward(self.grad.transpose())
                    self.creators[0].backward(self.grad.transpose(), self)

                if self.creation_op == 'mm':
                    act = self.creators[0]
                    weights = self.creators[1]
                    new = self.grad.mm(weights.transpose())
                    # act.backward(new)
                    act.backward(new, self)
                    new = self.grad.transpose().mm(act).transpose()
                    # weights.backward(new)
                    weights.backward(new, self)

                if self.creation_op == 'sigmoid':
                    ones = Tensor(np.ones_like(self.grad.data))
                    # self.creators[0].backward(self.grad * (self * (ones - self)))
                    self.creators[0].backward(self.grad * (self * (ones - self)), self)

                if self.creation_op == 'tanh':
                    ones = Tensor(np.ones_like(self.grad.data))
                    # self.creators[0].backward(self.grad * (ones - (self * self)))
                    self.creators[0].backward(self.grad * (ones - (self * self)), self)

                if self.creation_op == 'index_select':
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    # self.creators[0].backward(Tensor(new_grad))
                    self.creators[0].backward(Tensor(new_grad), self)

                if self.creation_op == 'cross_entropy':
                    dx = self.softmax_output - self.target_dist
                    # self.creators[0].backward(Tensor(dx))
                    self.creators[0].backward(Tensor(dx), self)

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

    def softmax(self):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape) - 1,
                                       keepdims=True)
        return softmax_output

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


class LSTMCell(Layer):
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

    def forward(self, input, hidden):
        prev_hidden = hidden[0]
        prev_cell = hidden[1]

        f = (self.xf.forward(input) + self.hf.forward(prev_hidden)).sigmoid()
        i = (self.xi.forward(input) + self.hi.forward(prev_hidden)).sigmoid()
        o = (self.xo.forward(input) + self.ho.forward(prev_hidden)).sigmoid()
        g = (self.xc.forward(input) + self.hc.forward(prev_hidden)).tanh()
        cell = (f * prev_cell) + (i * g)
        h = o * cell.tanh()

        output = self.w_ho.forward(h)
        return output, (h, cell)

    def init_hidden(self, batch_size=1):
        h = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        c = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        h.data[:, 0] += 1
        c.data[:, 0] += 1

        return (h, c)


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


with open('../data/shakespear.txt', 'r') as f:
    raw = f.read()

vocab = list(set(raw))
word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i
indices = np.array(list(map(lambda x: word2index[x], raw)))

embed = Embedding(vocab_size=len(vocab), dim=512)
model = LSTMCell(n_inputs=512, n_hidden=512, n_output=len(vocab))
model.w_ho.weight.data *= 0

criterion = CrossEntropyLoss()
optim = SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.01)

batch_size = 16
bptt = 25
n_batches = int((indices.shape[0] / batch_size))
trimmed_indices = indices[:n_batches*batch_size]
# batch_indices: each column represents a sub-sequence from indices -> continuous
batched_indices = trimmed_indices.reshape(batch_size, n_batches)
batched_indices = batched_indices.transpose()

input_batched_indices = batched_indices[:-1]
target_batched_indices = batched_indices[1:]

n_bptt = int((n_batches - 1) / bptt)
input_batches = input_batched_indices[:n_bptt*bptt]
input_batches = input_batches.reshape(n_bptt, bptt, batch_size)
target_batches = target_batched_indices[:n_bptt*bptt]
target_batches = target_batches.reshape(n_bptt, bptt, batch_size)

print('indices shape: {}'.format(indices.shape))
print('number batches: {}'.format(n_batches))
print('trimmed indices shape: {}'.format(trimmed_indices.shape))
print('batched indices shape: {}'.format(batched_indices.shape))
print('input batched indices: {}'.format(input_batched_indices.shape))
print('target batched indices: {}'.format(target_batched_indices.shape))
print('input batches shape: {}'.format(input_batches.shape))
print('target batches shape: {}'.format(target_batches.shape))

print('input batch 0: {}'.format(input_batches[0, :, 0]))
print('target batch 0: {}'.format(target_batches[0, :, 0]))
print('raw indices: {}'.format(indices[:16]))


def train(iterations=100):
    min_loss = 1000
    for iter in range(iterations):
        total_loss = 0
        n_loss = 0

        hidden = model.init_hidden(batch_size=batch_size)
        for batch_i in range(len(input_batches)):
            hidden = (Tensor(hidden[0].data, autograd=True),
                      Tensor(hidden[1].data, autograd=True))

            losses = list()
            for t in range(bptt):
                # print('t: {}'.format(t))
                input = Tensor(input_batches[batch_i][t], autograd=True)
                # print('input shape: {}'.format(input.data.shape))
                rnn_input = embed.forward(input=input)
                # print('embedded shape: {}'.format(rnn_input.data.shape))
                output, hidden = model.forward(input=rnn_input,
                                               hidden=hidden)
                # print('hidden shape: {}'.format(hidden.data.shape))
                # print('output shape: {}'.format(output.data.shape))

                target = Tensor(target_batches[batch_i][t], autograd=True)
                batch_loss = criterion.forward(output, target)
                # batch_loss.backward() # delete

                if t == 0:
                    losses.append(batch_loss)
                else:
                    losses.append(batch_loss + losses[-1])

            loss = losses[-1]
            loss.backward()
            optim.step()
            total_loss += loss.data / bptt
            epoch_loss = np.exp(total_loss / (batch_i + 1))

            if epoch_loss < min_loss:
                min_loss = epoch_loss
                print()

            log = '\r Iter:{}'.format(iter)
            log += ' - Alpha: {}'.format(optim.alpha)
            log += ' - Batch: {}/{}'.format(batch_i+1, len(input_batches))
            log += ' - Min Loss: {}'.format(min_loss)
            log += ' - Loss: {}'.format(epoch_loss)
            if batch_i % 1 == 0:
                sys.stdout.write(log)

        optim.alpha *= 0.99


train(100)


def generate_sample(n=30, init_char=' '):
    s = ' '
    hidden = model.init_hidden(batch_size=1)
    input = Tensor(np.array([word2index[init_char]]))
    for i in range(n):
        rnn_input = embed.forward(input)
        output, hidden = model.forward(input=rnn_input,
                                       hidden=hidden)
        output.data *= 10
        temp_dist = output.softmax()
        temp_dist /= temp_dist.sum()

        m = (temp_dist > np.random.rand()).argmax()
        c = vocab[m]
        input = Tensor(np.array([m]))
        s += c
    return s


print(generate_sample(n=2000, init_char='\n'))

