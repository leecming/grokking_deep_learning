import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))
relu = lambda x: (x > 0).astype(float) * x

weights = np.array([[1, 4],
                    [4, 1]])
activation = sigmoid(np.array([1, 0.01]))

print('Sigmoid Activations')
activations = list()
for _ in range(10):
    activation = sigmoid(activation.dot(weights))
    activations.append(activation)
    print(activation)
print('Sigmoid Gradients')
gradient = np.ones_like(activation)
for activation in reversed(activations):
    gradient = activation * (1 - activation) * gradient
    gradient = gradient.dot(weights.transpose())
    print(gradient)

print('Relu Activations')
print('Relu Activations')
activations = list()
for _ in range(10):
    activation = relu(activation.dot(weights))
    activations.append(activation)
    print(activation)
print('Relu Gradients')
gradient = np.ones_like(activation)
for activation in reversed(activations):
    gradient = ((activation > 0) * gradient).dot(weights.transpose())
    print(gradient)
