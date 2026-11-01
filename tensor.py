import numpy as np

DTYPE = "float32"
_grad_enabled = True


class Tensor:
    def __init__(self, data, _prev=(), _op="", requires_grad=False):
        if isinstance(data, Tensor):
            raise TypeError("Tensor does not accept Tensor as data.")

        self.data = np.array(data, dtype=DTYPE)
        self.grad = np.zeros(self.shape, dtype=DTYPE)
        self._prev = _prev if _grad_enabled else ()
        self._op = _op
        self.requires_grad = requires_grad if _grad_enabled else False
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @classmethod
    def zeros(cls, shape: tuple[int, ...], requires_grad=False):
        return cls(np.zeros(shape), requires_grad=requires_grad)

    @classmethod
    def ones(cls, shape: tuple[int, ...], requires_grad=False):
        return cls(np.ones(shape), requires_grad=requires_grad)

    @classmethod
    def normal(cls, shape: tuple[int, ...], std=1.0, loc=0.0, requires_grad=False):
        return cls(np.random.normal(loc=loc, scale=std, size=shape), requires_grad=requires_grad)

    def __repr__(self):
        return f"Tensor shape {self.shape} op {self._op} requires grad {self.requires_grad}"

    # Every operation below builds its output tensor and registers a _backward
    # closure that pushes out.grad to the inputs (chain rule).

    def __add__(self, other):
        other = _to_tensor(other)
        data = self.data + other.data
        out = Tensor(data=data, _prev=(self, other), _op="add", requires_grad=True)

        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)

        return out._attach(_backward)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        data = -self.data
        out = Tensor(data=data, _prev=(self,), _op="neg", requires_grad=True)

        def _backward():
            self.grad -= _unbroadcast(out.grad, self.shape)

        return out._attach(_backward)

    def __sub__(self, other):
        other = _to_tensor(other)
        data = self.data - other.data
        out = Tensor(data=data, _prev=(self, other), _op="sub", requires_grad=True)

        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad -= _unbroadcast(out.grad, other.shape)

        return out._attach(_backward)

    def __rsub__(self, other):
        other = _to_tensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = _to_tensor(other)
        data = self.data * other.data
        out = Tensor(data=data, _prev=(self, other), _op="mul", requires_grad=True)

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.shape)

        return out._attach(_backward)

    def __rmul__(self, other):
        other = _to_tensor(other)
        return other.__mul__(self)

    def __truediv__(self, other):
        other = _to_tensor(other)
        data = self.data / other.data
        out = Tensor(data=data, _prev=(self, other), _op="truediv", requires_grad=True)

        def _backward():
            self.grad += _unbroadcast(out.grad / other.data, self.shape)
            other.grad += _unbroadcast(-out.grad * self.data / (other.data**2), other.shape)

        return out._attach(_backward)

    def __rtruediv__(self, other):
        other = _to_tensor(other)
        return other.__truediv__(self)

    def __pow__(self, other):  # other must be a plain number
        if isinstance(other, Tensor):
            raise TypeError("Exponent can not be of type Tensor")

        data = self.data**other
        out = Tensor(data=data, _prev=(self,), _op="pow", requires_grad=True)

        def _backward():
            self.grad += _unbroadcast(other * out.grad * (self.data ** (other - 1)), self.shape)

        return out._attach(_backward)

    def __matmul__(self, other):
        other = _to_tensor(other)
        data = self.data @ other.data
        out = Tensor(data=data, _prev=(self, other), _op="matmul", requires_grad=True)

        def _backward():
            self.grad += _unbroadcast(out.grad @ other.data.swapaxes(-2, -1), self.shape)
            other.grad += _unbroadcast(self.data.swapaxes(-2, -1) @ out.grad, other.shape)

        return out._attach(_backward)

    def __rmatmul__(self, other):
        other = _to_tensor(other)
        return other.__matmul__(self)

    def __getitem__(self, ids):
        data = self.data[ids]
        out = Tensor(data=data, _prev=(self, ), _op="getitem", requires_grad=True)

        def _backward():
            np.add.at(self.grad, ids, out.grad)
        
        return out._attach(_backward)

    def exp(self):
        data = np.exp(self.data)
        out = Tensor(data=data, _prev=(self,), _op="exp", requires_grad=True)

        def _backward():
            self.grad += _unbroadcast(out.grad * data, self.shape)

        return out._attach(_backward)

    def log(self):
        data = np.log(self.data)
        out = Tensor(data=data, _prev=(self,), _op="log", requires_grad=True)

        def _backward():
            self.grad += _unbroadcast(out.grad / self.data, self.shape)

        return out._attach(_backward)

    def relu(self):
        data = np.maximum(self.data, 0)
        out = Tensor(data=data, _prev=(self,), _op="relu", requires_grad=True)

        def _backward():
            self.grad += _unbroadcast(out.grad * (data > 0), self.shape)

        return out._attach(_backward)

    def tanh(self):
        data = np.tanh(self.data)
        out = Tensor(data=data, _prev=(self,), _op="tanh", requires_grad=True)

        def _backward():
            self.grad += _unbroadcast(out.grad * (1 - (data**2)), self.shape)

        return out._attach(_backward)

    def sum(self, axis=None, keepdims=False):
        data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(data=data, _prev=(self,), _op="sum", requires_grad=True)

        def _backward():
            # Restore the reduced axis so the gradient broadcasts back
            if keepdims == False and axis is not None:
                self.grad += np.expand_dims(out.grad, axis=axis)
            else:
                self.grad += out.grad

        return out._attach(_backward)

    def mean(self, axis=None, keepdims=False):
        data = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(data=data, _prev=(self,), _op="mean", requires_grad=True)

        def _backward():
            # Each input contributes 1 / (number of elements averaged over)
            if keepdims == False and axis is not None:
                self.grad += np.expand_dims(out.grad / (self.data.size / out.grad.size), axis=axis)
            else:
                self.grad += out.grad / (self.data.size / out.grad.size)

        return out._attach(_backward)

    def max(self, axis=None, keepdims=False):
        # Not differentiable here, only used to stabilise softmax
        data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(data=data, _prev=(), _op="max", requires_grad=False)

        return out

    def reshape(self, *shape):
        data = self.data.reshape(shape)
        out = Tensor(data=data, _prev=(self,), _op="reshape", requires_grad=True)

        def _backward():
            self.grad += out.grad.reshape(self.shape)

        return out._attach(_backward)

    def transpose(self, axes=None):
        data = np.transpose(self.data, axes=axes)
        out = Tensor(data=data, _prev=(self,), _op="transpose", requires_grad=True)

        def _backward():
            if axes is None:
                self.grad += np.transpose(out.grad)
            else:
                # Undo the permutation
                self.grad += np.transpose(out.grad, np.argsort(axes))

        return out._attach(_backward)

    def _attach(self, fn):
        # Leaf tensors and tensors without grad keep the default no-op backward
        if self._prev == () or not self.requires_grad:
            return self
        else:
            self._backward = fn
            return self

    def item(self):
        return self.data.item()

    # works but might hit recursion limit better to use iterative TO
    def recursive_topological_order(self): 
        visited = set()
        topological_order = []

        def visit(node):
            for previous in node._prev:
                if previous not in visited:
                    visited.add(previous)
                    visit(previous)

            topological_order.append(node)

        visit(self)
        return topological_order

    # Iterative version to avoid recursion limit
    def topological_order(self):
        stack = [(self, False)]
        topological_order = []
        visited = set()

        while stack:
            node, state = stack.pop()
            if state:
                topological_order.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                for parent in node._prev:
                    if parent not in visited:
                        stack.append((parent, False))

        return topological_order

    def backward(self):
        if self.data.size != 1:
            raise ValueError("Start musst be scalar")

        order = self.topological_order()
        self.grad = np.ones(self.shape, dtype=DTYPE)

        # Walk from the output back to the inputs
        for node in reversed(order):
            node._backward()

    def softmax(self, axis=-1):
        # Subtracting the max avoids overflow in exp without changing the result
        norm = self - self.max(axis=axis, keepdims=True)
        exp_vals = norm.exp()
        return exp_vals / exp_vals.sum(axis=axis, keepdims=True)

    def log_softmax(self, axis=-1):
        norm = self - self.max(axis=axis, keepdims=True)
        exp_sum = norm.exp().sum(axis=axis, keepdims=True)

        return norm - exp_sum.log()


class Parameter(Tensor):
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=True)


# Context manager that disables graph building for evaluation
class NoGrad:
    def __init__(self):
        pass

    def __enter__(self):
        global _grad_enabled
        self.prev_value = _grad_enabled
        _grad_enabled = False

    def __exit__(self, exc_type, exc, tb):
        global _grad_enabled
        _grad_enabled = self.prev_value


def _to_tensor(value):  # value can be a number, array or already a Tensor
    if isinstance(value, Tensor):
        return value
    else:
        return Tensor(value)


def _unbroadcast(grad_update, shape):
    # Sum a gradient back down to the shape the input had before broadcasting
    if grad_update.shape == shape:
        return grad_update

    # Sum away leading axes that broadcasting added
    if grad_update.ndim != len(shape):
        grad_update = np.sum(grad_update, axis=tuple(range(grad_update.ndim - len(shape))))

    # Sum over axes that were stretched from size 1
    for axis in range(grad_update.ndim):
        if shape[axis] == 1:
            grad_update = np.sum(grad_update, axis=axis, keepdims=True)

    return grad_update
