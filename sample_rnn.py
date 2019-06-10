import sys
import numpy as np
from tensor import Tensor
from layers import Embedding
from rnn import RNNCell
from losses import CrossEntropyLoss
from optimizers import SGD

with open('data/shakespear.txt', 'r') as f:
    raw = f.read()

vocab = list(set(raw))
word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i
indices = np.array(list(map(lambda x: word2index[x], raw)))

embed = Embedding(vocab_size=len(vocab), dim=512)
model = RNNCell(n_inputs=512, n_hidden=512, n_output=len(vocab))

criterion = CrossEntropyLoss()
optim = SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.01)

batch_size = 32
bptt = 16
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
    for iter in range(iterations):
        total_loss = 0

        hidden = model.init_hidden(batch_size=batch_size)
        for batch_i in range(len(input_batches)):
            hidden = Tensor(hidden.data, autograd=True)
            loss = None
            for t in range(bptt):
                input = Tensor(input_batches[batch_i][t], autograd=True)
                rnn_input = embed.forward(input)
                output, hidden = model.forward(rnn_input,
                                               hidden)

                target = Tensor(target_batches[batch_i][t], autograd=True)
                batch_loss = criterion.forward(output, target)

                if t == 0:
                    loss = batch_loss
                else:
                    loss = loss + batch_loss

            loss.backward()
            optim.step()
            total_loss += loss.data

            log = '\r Iter:{}'.format(iter)
            log += ' - Alpha: {}'.format(optim.alpha)
            log += ' - Batch: {}/{}'.format(batch_i+1, len(input_batches))
            log += ' - Loss: {}'.format(total_loss / (batch_i+1))
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
        output, hidden = model.forward(rnn_input,
                                       hidden)
        output.data *= 10
        temp_dist = output.softmax()
        temp_dist /= temp_dist.sum()

        m = (temp_dist > np.random.rand()).argmax()
        c = vocab[m]
        input = Tensor(np.array([m]))
        s += c
    return s


print(generate_sample(n=2000, init_char='\n'))