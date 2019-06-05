import numpy as np


def softmax(x_):
    x = np.atleast_2d(x_)
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


word_vects = {}
word_vects['yankees'] = np.array([0., 0., 0.])
word_vects['bears'] = np.array([0., 0., 0.])
word_vects['braves'] = np.array([0., 0., 0.])
word_vects['red'] = np.array([0., 0., 0.])
word_vects['sox'] = np.array([0., 0., 0.])
word_vects['lose'] = np.array([0., 0., 0.])
word_vects['defeat'] = np.array([0., 0., 0.])
word_vects['beat'] = np.array([0., 0., 0.])
word_vects['tie'] = np.array([0., 0., 0.])

sent2output = np.random.rand(3, len(word_vects))
identity = np.eye(3)

layer_0 = word_vects['red']
layer_1 = layer_0.dot(identity) + word_vects['sox']
layer_2 = layer_1.dot(identity) + word_vects['defeat']

pred = softmax(layer_2.dot(sent2output))

y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])

pred_delta = pred - y
layer_2_delta = pred_delta.dot(sent2output.T)
defeat_delta = layer_2_delta * 1
layer_1_delta = layer_2_delta.dot(identity.T)
sox_delta = layer_1_delta * 1
layer_0_delta = layer_1_delta.dot(identity.T)
alpha = 0.01
word_vects['red'] -= layer_0_delta * alpha
word_vects['sox'] -= sox_delta * alpha
word_vects['defeat'] -= defeat_delta * alpha
identity -= np.outer(layer_0, layer_1_delta) * alpha
identity -= np.outer(layer_1, layer_2_delta) * alpha
sent2output -= np.outer(layer_2, pred_delta) * alpha
