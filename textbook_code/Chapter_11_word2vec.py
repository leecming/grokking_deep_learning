import sys, random, math
from collections import Counter
import numpy as np
import pickle as pl

np.random.seed(1)
random.seed(1)
f = open('../data/reviews.txt')
raw_reviews = f.readlines()
f.close()

tokens = list(map(lambda x: (x.split(" ")), raw_reviews))
wordcnt = Counter()
for sent in tokens:
    for word in sent:
        wordcnt[word] += 1
vocab = list(set(map(lambda x: x[0], wordcnt.most_common())))

word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i
pl.dump(word2index, open('../data/word2index_imdb.p', 'wb'))

concatenated = list()
input_dataset = list()
for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
            sent_indices.append(word2index[word])
            concatenated.append(word2index[word])
        except:
            pass
    input_dataset.append(sent_indices)
concatenated = np.array(concatenated)
random.shuffle(input_dataset)

alpha, iterations = (0.05, 2)
hidden_size, window, negative = (50, 2, 5)

weights_0_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
weights_1_2 = np.random.rand(len(vocab), hidden_size) * 0

layer_2_target = np.zeros(negative + 1)
layer_2_target[0] = 1


def similar(target='beautiful'):
    target_index = word2index[target]

    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - (weights_0_1[target_index])
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))
    return scores.most_common(10)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for rev_i, review in enumerate(input_dataset * iterations):
    for target_i in range(len(review)):
        # since it's really expensive to predict every vocabulary
        # we're only going to predict a random subset
        target_samples = [review[target_i]] + list(concatenated \
                                                       [(np.random.rand(negative) * len(concatenated)).astype(
                'int').tolist()])

        left_context = review[max(0, target_i - window):target_i]
        right_context = review[target_i + 1:min(len(review), target_i + window)]

        layer_1 = np.mean(weights_0_1[left_context + right_context], axis=0)
        layer_2 = sigmoid(layer_1.dot(weights_1_2[target_samples].T))
        layer_2_delta = layer_2 - layer_2_target
        layer_1_delta = layer_2_delta.dot(weights_1_2[target_samples])

        weights_0_1[left_context + right_context] -= layer_1_delta * alpha
        weights_1_2[target_samples] -= np.outer(layer_2_delta, layer_1) * alpha

    if rev_i % 250 == 0:
        sys.stdout.write('\rProgress:' + str(rev_i / float(len(input_dataset)
                                                           * iterations)) + "   " + str(similar('terrible')))
    sys.stdout.write('\rProgress:' + str(rev_i / float(len(input_dataset)
                                                       * iterations)))
print(similar('terrible'))
np.save('../data/imdb_weights_0_1.npy', weights_0_1)
np.save('../data/imdb_weights_1_2.npy', weights_1_2)