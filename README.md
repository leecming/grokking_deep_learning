### Simple autograd framework

An autograd framework built from scratch, dependent only on numpy and referencing [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) material.

The core of the framework resides in the [Tensor](tensor.py) class which contains tensor data, supports tensor operations (e.g., addition, softmax etc.), and also handles backprop -
- Tensors are linked together via their children and parents attributes
- Forward propagation is handled by calling various operation methods (e.g., add) which computes the new tensor for the result of the operation, creates the child Tensor object representing that computation, and initializes the child with details of how it was created (e.g., parent tensors and the op that created it)
- Backprop is handled by the backward method where downstream Tensors flow gradients back to their parent Tensors 
- With tensor objects linked together by operations, each Tensor accumulates gradients calculated by their children, then calculates their parents gradients by reversing the op linking the parent(s) to them (typically the derivative of the computation) and hands it to them
- After backprop, the Tensors objects representing model parameters can be updated

Layer level abstractions can be found in 
- [activations](activations.py) (Sigmoid, Tanh)
- [layers](layers.py) module (Layer base class, Embedding, Linear, Sequential)
- [losses](losses.py) module (MSE, cross-entropy)
- [optimizers](optimizers.py) module (SGD)
- [rnn](rnn.py) module (basic RNN cell, LSTM cell)


Sample code
```
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
```

The following tensor ops are supported:
1. multiplication
2. addition
3. subtraction
4. negation
5. sum (across dimension)
6. expansion (creates a new dim, where the input is repeated by specified number) 
7. transposition
8. dot product
9. sigmoid activation
10. tanh activation
11. cross entropy
12. index selection  

#### Setup
1. A [DockerFile](Dockerfile) is provided which builds against a vanilla Tensorflow-1.13 (CPU) image, and installs the necessary system, and Python prerequisites - IMPORTANT: the Dockerfile installs an SSH server with a default password
2. Textbook code uses data (to be placed in the data/ subfolder) from: 
    - Babi (run: wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-1.tar.gz,  tar -xvf tasks_1-20_v1-1.tar.gz)
    - reviews.txt, labels.txt, shakespear.txt (from the [Grokking Deep Learning GitHub](https://github.com/iamtrask/Grokking-Deep-Learning))