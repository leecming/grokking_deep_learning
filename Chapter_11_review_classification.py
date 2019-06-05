import numpy as np
np.random.seed(1)

with open('data/reviews.txt') as f:
    raw_reviews = f.readlines()

with open('data/labels.txt') as f:
    raw_labels = f.readlines()

tokens = list(map(lambda x: set(x.split(' ')), raw_reviews))

vocab = set()
for sent in tokens:
    for word in sent:
        if len(word) > 0:
            vocab.add(word)

vocab = list(vocab)

word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i

input_dataset = list()
for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
            sent_indices.append(word2index[word])
        except:
            pass
    input_dataset.append(list(set(sent_indices)))

target_dataset = list()
for label in raw_labels:
    if label == 'positive\n':
        target_dataset.append(1)
    else:
        target_dataset.append(0)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


alpha = 0.01
iterations = 2
hidden_size = 100

weights_0_1 = 0.2 * np.random.random((len(vocab), hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1

correct = 0
total = 0

for iter in range(iterations):
    for i in range(len(input_dataset) - 1000):
        x, y = input_dataset[i], target_dataset[i]
        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

        layer_2_delta = layer_2 - y
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)

        weights_0_1[x] -= layer_1_delta * alpha
        weights_1_2 -= np.outer(layer_1, layer_2_delta) * alpha

        if np.abs(layer_2_delta) < 0.5:
            correct += 1
        total += 1
        if i % 10 == 0:
            progress = i / float(len(input_dataset))
            print('Iter: {}, Progress: {}, Acc: {}'.format(iter,
                                                           progress,
                                                           correct/float(total)))
            print()

correct = 0
total = 0

for i in range(len(input_dataset)-1000, len(input_dataset)):
    x, y = input_dataset[i], target_dataset[i]

    layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
    layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

    if np.abs(layer_2 - y) < 0.5:
        correct += 1
    total += 1
print('Test accuracy: {}'.format(correct/float(total)))
