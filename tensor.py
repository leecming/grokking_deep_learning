"""
Core autograd framework code resides within the Tensor class.
Tensor objects hold tensor data and are chained together by
the supported operations to provide a graph supporting
automatic forward propagation and backprop

Following ops supported:
1. multiplication
2. addition
3. subtraction
4. negation
5. summation (across dimension)
6. expansion (across dimension)
7. transposition
8. dot product
9. sigmoid activation
10. tanh activation
11. cross entropy
12. index selection
"""
from uuid import uuid4
import numpy as np


class Tensor:
    """
    Container for tensors
    - Chained together with other tensors and tracked with the
      children attrib (child id->count of gradients received), the parents
      attrib (list of parents) & the creation_op (str for how the tensor was created)
    - When the parent tensor(s) create a new tensor - they initialize the child tensor with
      references to themselves
      - provides a forward chain for forward prop
      - the child at init also updates the parents' children attributes with refs to itself
    - On backprop - a tensor calls its backward method (and when it has received all
      gradients from its children) flows its computed gradient to its parents by invoking their
      backward method
    """
    def __init__(self,
                 data,
                 autograd=False,
                 parents=None,
                 creation_op=None,
                 tensor_id=None):
        self.data = np.array(data)
        self.creation_op = creation_op
        self.parents = parents
        self.grad = None
        self.autograd = autograd
        self.children = {}
        self.tensor_id = uuid4() if tensor_id is None else tensor_id

        # add itself to creators' children mappings
        if parents is not None:
            for curr_parent in parents:
                if self.tensor_id not in curr_parent.children:
                    curr_parent.children[self.tensor_id] = 1
                else:
                    curr_parent.children[self.tensor_id] += 1

        # Used by the index select op
        self.index_select_indices = None
        # Used by the softmax op
        self.target_dist = None
        self.softmax_output = None

    def all_children_grads_accounted_for(self):
        """
        check tensor has received all gradients from every child
        """
        for _, count in self.children.items():
            if count != 0:
                return False
        return True

    def backward(self, grad=None, grad_origin=None):
        """
        Accumulates gradients from children, computes the gradients for its
        parents (i.e., pre-creation op) and flows them to its parent(s)
        :param grad: its gradient for a given child
        :param grad_origin: the child Tensor its receiving a gradient from
        """
        if self.autograd:
            if grad is None: # convenience function - if terminal tensor
                grad = Tensor(np.ones_like(self.data))

            # grad_origin may be None if current Tensor is terminal
            if grad_origin is not None:
                if self.children[grad_origin.tensor_id] == 0:
                    # should never get here
                    print('origin id: {}'.format(grad_origin.tensor_id))
                    print('self id: {}'.format(self.tensor_id))
                    print('self creation_op: {}'.format(self.creation_op))
                    print('# creators: {}'.format(len(self.parents)))
                    for curr_parent in self.parents:
                        print('Creator id {}, creation_op {}'.format(curr_parent.tensor_id,
                                                                     curr_parent.creation_op))
                    raise Exception('cannot backprop more than once')
                else:
                    # update that it has received a gradient from its child
                    self.children[grad_origin.tensor_id] -= 1

            # accumulate gradients
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad

            # only flow gradients back to parents if it has received ALL gradients from its children
            # look up its creation op, apply the gradient computation (typically
            # the derivative of the op itself) and call its parents' backward method
            # with the computed gradient
            if self.parents is not None and \
                    (self.all_children_grads_accounted_for() or grad_origin is None):
                if self.creation_op == 'add':
                    self.parents[0].backward(self.grad, self)
                    self.parents[1].backward(self.grad, self)

                if self.creation_op == 'neg':
                    # self.creators[0].backward(self.grad.__neg__())
                    self.parents[0].backward(self.grad.__neg__(), self)

                if self.creation_op == 'sub':
                    self.parents[0].backward(self.grad, self)
                    self.parents[1].backward(self.grad.__neg__(), self)

                if self.creation_op == 'mul':
                    new = self.grad * self.parents[1]
                    self.parents[0].backward(new, self)
                    new = self.grad * self.parents[0]
                    self.parents[1].backward(new, self)

                if "sum" in self.creation_op:
                    dim = int(self.creation_op.split("_")[1])
                    self.parents[0].backward(self.grad.expand(dim,
                                                              self.parents[0].data.shape[dim]),
                                             self)

                if 'expand' in self.creation_op:
                    dim = int(self.creation_op.split('_')[1])
                    self.parents[0].backward(self.grad.sum(dim), self)

                if self.creation_op == 'transpose':
                    self.parents[0].backward(self.grad.transpose(), self)

                if self.creation_op == 'dot':
                    act = self.parents[0]
                    weights = self.parents[1]
                    new = self.grad.dot(weights.transpose())
                    act.backward(new, self)
                    new = self.grad.transpose().dot(act).transpose()
                    weights.backward(new, self)

                if self.creation_op == 'sigmoid':
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.parents[0].backward(self.grad * (self * (ones - self)), self)

                if self.creation_op == 'tanh':
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.parents[0].backward(self.grad * (ones - (self * self)), self)

                if self.creation_op == 'index_select':
                    new_grad = np.zeros(self.parents[0].data.shape)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.parents[0].backward(Tensor(new_grad), self)

                if self.creation_op == 'cross_entropy':
                    new_grad = self.softmax_output - self.target_dist
                    self.parents[0].backward(Tensor(new_grad), self)

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd=True,
                          parents=[self, other],
                          creation_op='add')
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                          autograd=True,
                          parents=[self, other],
                          creation_op='sub')
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                          autograd=True,
                          parents=[self, other],
                          creation_op='mul')
        return Tensor(self.data * other.data)

    def sum(self, dim):
        """
        Sum across specified dim
        :param dim:
        :return:
        """
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          parents=[self],
                          creation_op='sum_{}'.format(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        """
        Expand (repeat itself specified num of times) across the specified dimension
        - used in backprop for a sum op
        """
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_data = self.data.repeat(copies)
        new_data = new_data.reshape(list(self.data.shape)+[copies]).transpose(trans_cmd)

        if self.autograd:
            return Tensor(new_data,
                          autograd=True,
                          parents=[self],
                          creation_op='expand_{}'.format(dim))
        return Tensor(new_data)

    def transpose(self):
        """ Vanilla transpose op"""
        if self.autograd:
            return Tensor(self.data.transpose(),
                          autograd=True,
                          parents=[self],
                          creation_op='transpose')
        return Tensor(self.data.transpose())

    def dot(self, other):
        """ Vanila dot product """
        if self.autograd:
            return Tensor(self.data.dot(other.data),
                          autograd=True,
                          parents=[self, other],
                          creation_op='dot')
        return Tensor(self.data.dot(other.data))

    def __neg__(self):
        """ Negation """
        if self.autograd:
            return Tensor(self.data * -1,
                          autograd=True,
                          parents=[self],
                          creation_op='neg')
        return Tensor(self.data * -1)

    def sigmoid(self):
        """ Sigmoid op """
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          parents=[self],
                          creation_op='sigmoid')
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        """ Tanh op"""
        if self.autograd:
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          parents=[self],
                          creation_op='tanh')
        return Tensor(np.tanh(self.data))

    def softmax(self):
        """ Softmax op """
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape) - 1,
                                       keepdims=True)
        return softmax_output

    def index_select(self, indices):
        """ Select tensor data by specified indices e.g., embedding """
        if self.autograd:
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         parents=[self],
                         creation_op='index_select')
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])

    def cross_entropy(self, target_indices):
        """
        Note it sets the softmax_output and target_dist
        attributes for the child tensor used for backprop
        """
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape)-1,
                                       keepdims=True)
        flattened_target_indices = target_indices.data.flatten()
        probs = softmax_output.reshape(len(flattened_target_indices), -1)
        target_dist = np.eye(probs.shape[1])[flattened_target_indices]
        loss = - (np.log(probs) * target_dist).sum(1).mean()

        if self.autograd:
            out = Tensor(loss,
                         autograd=True,
                         parents=[self],
                         creation_op='cross_entropy')
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out
        return Tensor(loss)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())
