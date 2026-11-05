import numpy as np


# Base class that stores the parameters to update. Subclasses implement step.
class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    # Gradients accumulate across backward calls, so reset them before each step.
    # zeros_like keeps the dtype of each parameter instead of reading tensor.DTYPE.
    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = np.zeros_like(parameter.data)


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.1):
        super().__init__(parameters, lr)

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad


# Adam keeps a running mean of the gradients (m) and of their squares (v)
# and scales each update by them
class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # step counter for bias correction
        self.m = [np.zeros_like(parameter.data) for parameter in self.parameters]
        self.v = [np.zeros_like(parameter.data) for parameter in self.parameters]

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


# Scales all gradients down together if their combined L2 norm exceeds max_norm.
# Returns the norm before clipping, which is useful for logging.
def clip_grad_norm(parameters, max_norm=1):
    param_sum = 0
    for parameter in parameters:
        param_sum += np.sum(parameter.grad**2)

    norm = np.sqrt(param_sum)

    if norm > max_norm:
        for parameter in parameters:
            # The small epsilon avoids division by zero
            parameter.grad = (max_norm / (norm + 1e-6)) * parameter.grad

    return norm
