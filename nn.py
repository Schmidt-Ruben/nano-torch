import numpy as np
from tensor import Parameter, Tensor


class Module:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError("Module has no forward Method")

    def __call__(self, *args):
        return self.forward(*args)

    def parameters(self):
        # Collect all Parameters also from nested modules without duplicates
        parameters = []

        def check_attribute(attribute):
            if isinstance(attribute, Module):
                new_params = [param for param in attribute.parameters() if param not in parameters]
                parameters.extend(new_params)
            elif isinstance(attribute, Parameter):
                if attribute not in parameters:
                    parameters.append(attribute)
            elif isinstance(attribute, (list, tuple)):
                for item in attribute:
                    check_attribute(item)

        for attribute in vars(self).values():
            check_attribute(attribute)

        return parameters

    def num_params(self):
        return sum([parameter.data.size for parameter in self.parameters()])


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, std=None):
        super().__init__()
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        # He initialisation with variance 2 / fan_in
        std = (2 / in_features) ** (1 / 2) if std is None else std
        self.W = Parameter.normal(
            (in_features, out_features), std=std, requires_grad=True
        )
        self.b = Parameter.zeros((out_features,), requires_grad=True) if bias else None

    def forward(self, x):
        if self.bias:
            return x @ self.W + self.b
        else:
            return x @ self.W


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = list(modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

class LayerNorm(Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.d = d
        self.eps = eps
        self.gamma = Parameter.ones((self.d,))
        self.beta = Parameter.zeros((self.d,))
    
    def forward(self, x: Tensor):
        mean = x.mean(axis=-1, keepdims=True)
        variance = ((x - mean)**2).mean(axis=-1, keepdims=True)
        normalised = (x - mean) / (variance + self.eps)**(1/2)
        return self.gamma * normalised + self.beta

class GELU(Module):
    def forward(self, x): # Standard Tanh approximation instead of erf 
        return 1/2 * x * (1 + Tensor.tanh((np.sqrt(2/np.pi))*(x + 0.044715 * x**3)))

class FeedForward(Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.fc1 = Linear(d, 4*d)
        self.act = GELU()
        self.fc2 = Linear(4*d, d)

    def forward(self, x):
        a = self.fc1.forward(x)
        b = self.act.forward(a)
        return self.fc2.forward(b) 

class MultiHeadAttention(Module):
    def __init__(self, d, n_heads, block_size):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError("d and n_heads are not compatible. Makes sure d is a multiple of n_heads.")
        self.d = d
        self.n_heads = n_heads
        self.block_size = block_size
        self.head_size = int(d/n_heads)
        self.q_map = Linear(in_features=d, out_features=d)
        self.k_map = Linear(in_features=d, out_features=d)
        self.v_map = Linear(in_features=d, out_features=d)
        self.out_proj = Linear(in_features=d, out_features=d)
        self.mask = np.where(np.tri(block_size, block_size, dtype='bool'), 0.0, -np.inf)

    def forward(self, x):
        Q = self.q_map(x)
        K = self.k_map(x)
        V = self.v_map(x)

        Q_reshape = Q.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_size)
        K_reshape = K.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_size)
        V_reshape = V.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_size)

        Q_reshape_t = Q_reshape.transpose(axes=(0, 2, 1, 3))
        K_reshape_t = K_reshape.transpose(axes=(0, 2, 1, 3))
        V_reshape_t = V_reshape.transpose(axes=(0, 2, 1, 3))

        S = Q_reshape_t @ K_reshape_t.transpose(axes=(0, 1, 3, 2)) / np.sqrt(self.head_size)
        S_masked = S + self.mask[:x.shape[1], :x.shape[1]]

        A = S_masked.softmax()
        O = A @ V_reshape_t
        O_t = O.transpose(axes=(0, 2, 1, 3))
        O_cat = O_t.reshape(x.shape[0], x.shape[1], self.d)

        Y = self.out_proj(O_cat)
        return Y

class Block(Module):
    def __init__(self, d, n_heads, block_size):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.block_size = block_size
        self.ln1 = LayerNorm(d)
        self.attn = MultiHeadAttention(d=d, n_heads=n_heads, block_size=block_size)
        self.ln2 = LayerNorm(d)
        self.ff = FeedForward(d)

    def forward(self, x):
        A = x + self.attn(self.ln1(x))
        B = A + self.ff(self.ln2(A))

        return B


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Parameter.normal((num_embeddings, embedding_dim), std=0.02)

    def forward(self, ids):
        if ids.dtype != 'int64':
            raise TypeError('ids must be dtype int64.')
        return self.weight[ids]


def one_hot_encoding(array, n_labels):    
    encoded_labels = np.zeros((array.size, n_labels), dtype=int)
    encoded_labels[np.arange(array.size), array] = 1
    return encoded_labels


def cross_entropy(logits, targets):
    n_labels = logits.shape[-1]
    if isinstance(targets, Tensor):
        targets = targets.data

    if logits.shape[:-1] != targets.shape:
        raise ValueError("Array and Labels don't have compatible shapes.")
    elif not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("Labels must be dtype int64.")
    elif targets.max() >=  n_labels or targets.min() < 0:
        raise ValueError("Labels must all be valid indices.")

    reindexed_targets = targets.reshape(-1)
    reindexed_logits = logits.reshape(-1, logits.shape[-1])
    N = reindexed_logits.shape[0]
    
    return -1 * (reindexed_logits.log_softmax()[np.arange(N), reindexed_targets]).mean()

