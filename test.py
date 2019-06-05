import math
import numpy as np
from collections import Counter

with open('data/reviews.txt') as f:
    raw_reviews = f.readlines()

tokens = list(map(lambda x: (x.split(" ")), raw_reviews))
wordcnt = Counter()
for sent in tokens:
    for word in sent:
        wordcnt[word] += 1
vocab = list(set(map(lambda x: x[0], wordcnt.most_common())))

print(vocab[:10])