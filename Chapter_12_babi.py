import sys
import random
import math
from collections import Counter
import numpy as np

with open('data/tasksv11/en/qa1_single-supporting-fact_train.txt', 'r') as f:
    raw = f.readlines()

tokens = list()
for line in raw[:1000]:
    tokens.append(line.lower().replace('\n', '').split(' ')[1:])

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


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


np.random.seed(1)
embed_size = 10

embed = (np.random.rand(len(vocab), embed_size) - 0.5) * 0.1
recurrent = np.eye(embed_size)
start = np.zeros(embed_size)
decoder = (np.random.rand(embed_size, len(vocab)) - 0.5) * 0.1
one_hot = np.eye(len(vocab))


def predict(sent):
    layers = list()
    layer = dict()
    layer['hidden'] = start
    layers.append(layer)

    loss = 0

    for target_i in range(len(sent)):
        layer = dict()
        layer['pred'] = softmax(layers[-1]['hidden'].dot(decoder))
        loss += -np.log(layer['pred'][sent[target_i]])
        layer['hidden'] = layers[-1]['hidden'].dot(recurrent) + embed[sent[target_i]]
        layers.append(layer)
    return layers, loss


for iter in range(30000):
    alpha = 0.001
    sent = words2indices(tokens[iter % len(tokens)][1:])
    layers, loss = predict(sent)

    for layer_idx in reversed(range(len(layers))):
        layer = layers[layer_idx]
        target = sent[layer_idx-1]

        if layer_idx > 0:
            layer['output_delta'] = layer['pred'] - one_hot[target]
            new_hidden_delta = layer['output_delta'].dot(decoder.T)

            if layer_idx == len(layers) - 1:
                layer['hidden_delta'] = new_hidden_delta
            else:
                layer['hidden_delta'] = new_hidden_delta + layers[layer_idx+1]['hidden_delta'].dot(recurrent.T)
        else:
            layer['hidden_delta'] = layers[layer_idx+1]['hidden_delta'].dot(recurrent.T)

    start -= layers[0]['hidden_delta'] * alpha / float(len(sent))
    for layer_idx, layer in enumerate(layers[1:]):
        decoder -= np.outer(layers[layer_idx]['hidden'], layer['output_delta']) * alpha / float(len(sent))
        embed_idx = sent[layer_idx]
        embed[embed_idx] -= layers[layer_idx]['hidden_delta'] * alpha/float(len(sent))
        recurrent -= np.outer(layers[layer_idx]['hidden'], layer['hidden_delta']) * alpha / float(len(sent))

    if iter % 1000 == 0:
        print('Perplexity: {}'.format(np.exp(loss/len(sent))))

sent_index = 4
l, _ = predict(words2indices(tokens[sent_index]))

print(tokens[sent_index])

for i,each_layer in enumerate(l[1:-1]):
    input = tokens[sent_index][i]
    true = tokens[sent_index][i+1]
    pred = vocab[each_layer['pred'].argmax()]
    print("Prev Input:" + input + (' ' * (12 - len(input))) + "True:" + true + (" " * (15 - len(true))) + "Pred:" + pred)
