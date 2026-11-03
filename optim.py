import numpy as np
from tensor import DTYPE


class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = np.zeros(shape=parameter.shape, dtype=DTYPE)


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.1):
        super().__init__(parameters, lr)

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad


class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # step counter for bias correction
        self.m = [np.zeros(parameter.shape, dtype=DTYPE) for parameter in self.parameters]
        self.v = [np.zeros(parameter.shape, dtype=DTYPE) for parameter in self.parameters]

    def step(self):
        self.t += 1

        for index in range(len(self.parameters)):
            self.m[index] = (
                self.beta1 * self.m[index] + (1 - self.beta1) * self.parameters[index].grad
            )
            self.v[index] = (
                self.beta2 * self.v[index] + (1 - self.beta2) * (self.parameters[index].grad**2)
            )

            # Bias correction for the zero-initialised moments
            m_hat = self.m[index] / (1 - self.beta1**self.t)
            v_hat = self.v[index] / (1 - self.beta2**self.t)

            self.parameters[index].data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_grad_norm(parameters, max_norm=1):
    param_sum = 0
    for parameter in parameters:
        param_sum += np.sum(parameter.grad**2)

    norm = np.sqrt(param_sum)

    if norm > max_norm:
        for parameter in parameters:
            parameter.grad = (max_norm / (norm + 1e-6)) * parameter.grad #avoiding div by zero

    return norm