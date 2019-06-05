import math
import numpy as np
from collections import Counter
import pickle as pl

with open('data/reviews.txt') as f:
    raw_reviews = f.readlines()

word2index = pl.load(open('data/word2index_imdb.p', 'rb'))
weights_0_1 = np.load('data/imdb_weights_0_1.npy', allow_pickle=True)


def analogy(positive=['terrible', 'good'], negative=['bad']):
    norms = np.sum(weights_0_1 * weights_0_1, axis=1) ** 0.5
    norms.resize(norms.shape[0], 1)
    normed_weights = weights_0_1 * norms

    query_vect = np.zeros(len(weights_0_1[0]))
    for word in positive:
        query_vect += normed_weights[word2index[word]]

    for word in negative:
        query_vect -= normed_weights[word2index[word]]

    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - query_vect
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))

    return scores.most_common(10)[1:]


print(analogy(['elizabeth', 'he'], ['she']))
