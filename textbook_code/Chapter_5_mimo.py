# toes, win, fans
weights = [[0.1, 0.1, -0.3],  # hurt
           [0.1, 0.2, 0.0],   # win
           [0.0, 1.3, 0.1]]   # sad


def w_sum(a, b):
    assert len(a) == len(b)
    output = 0
    for i in range(len(a)):
        output += a[i] * b[i]
    return output


def vect_mat_mul(vect, matrix):
    assert len(vect) == len(matrix)
    output = [0, 0, 0]
    for i in range(len(vect)):
        output[i] = w_sum(vect, matrix[i])
    return output


def neural_network(input, weights):
    pred = vect_mat_mul(input, weights)
    return pred


def zeros_matrix(n_rows, n_cols):
    return [[0] * n_cols] * n_rows


def outer_prod(vec_a, vec_b):
    out = zeros_matrix(len(vec_a), len(vec_b))

    for i in range(len(vec_a)):
        for j in range(len(vec_b)):
            out[i][j] = vec_a[i] * vec_b[j]

    return out


toes = [8.5, ]
wlrec = [0.65, ]
nfans = [1.2, ]

hurt = [0.1, ]
win = [1, ]
sad = [0.1, ]

alpha = 0.01

input = [toes[0], wlrec[0], nfans[0]]
true = [hurt[0], win[0], sad[0]]

for iter in range(3):
    pred = neural_network(input, weights)

    error = [0, 0, 0]
    delta = [0, 0, 0]

    for i in range(len(true)):
        error[i] = (pred[i] - true[i]) ** 2
        delta[i] = pred[i] - true[i]

    weight_deltas = outer_prod(input, delta)

    for i in range(len(weights)):
        for j in range(len(weights[0])):
            weights[i][j] -= alpha * weight_deltas[i][j]

    print(error)
