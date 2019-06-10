weights = [0.3, 0.2, 0.9]


def ele_mul(scalar, vector):
    out = [0, 0, 0]
    for i in range(len(out)):
        out[i] = vector[i] * scalar
    return out


def neural_network(input, weights):
    pred = ele_mul(input, weights)
    return pred

wlrec = [0.65, 1.0, 1.0, 0.9]
hurt = [0.1, 0.0, 0.0, 0.1]
win = [1, 1, 0, 1]
sad = [0.1, 0.0, 0.1, 0.2]

input = wlrec[0]
true = [hurt[0], win[0], sad[0]]

for iter in range(3):
    pred = neural_network(input, weights)

    error = [0, 0, 0]
    delta = [0, 0, 0]

    for i in range(len(true)):
        error[i] = (pred[i] - true[i])**2
        delta[i] = pred[i] - true[i]

    weight_deltas = ele_mul(input, delta)
    alpha = 0.1

    for i in range(len(weights)):
        weights[i] -= weight_deltas[i] * alpha

    print('Iter: {}'.format(iter))
    print('Weights: {}'.format(weights))
    print('Weight deltas: {}'.format(weight_deltas))
    print('Error: {}'.format(error))