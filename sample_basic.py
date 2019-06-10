import numpy as np
from tensor import Tensor
from layers import Sequential, Linear
from activations import Tanh, Sigmoid
from optimizers import SGD
from losses import MSELoss

np.random.seed(0)

data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), autograd=True)
target = Tensor(np.array([[0], [1], [0], [1]]), autograd=True)

model = Sequential([Linear(2, 3), Tanh(), Linear(3, 1), Sigmoid()])
criterion = MSELoss()

optim = SGD(parameters=model.get_parameters(), alpha=1)

for i in range(10):
    pred = model.forward(data)
    loss = criterion.forward(pred, target)

    loss.backward()
    optim.step()
    print(loss)
