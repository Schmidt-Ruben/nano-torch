import numpy as np
from tensor import Parameter, Tensor, NoGrad, get_rng

# Global switch that EvalMode turns off. Dropout is only active while it is True.
_training_enabled = True


# Base class for all layers and models. Subclasses implement forward.
class Module:
    def __init__(self):
        pass

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


# Fully connected layer computing x @ W + b
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


# Runs the given modules one after another
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = list(modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


# Normalises each vector over the last axis, then applies a learned scale and shift
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
        normalised = (x - mean) / (variance + self.eps)**(1 / 2)
        return self.gamma * normalised + self.beta


class GELU(Module):
    def forward(self, x):  # Standard tanh approximation instead of erf
        return 1 / 2 * x * (1 + Tensor.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# Position wise MLP of a transformer block. Expands to 4 * d and projects back.
class FeedForward(Module):
    def __init__(self, d, dropout=0.0):
        super().__init__()
        self.d = d
        self.fc1 = Linear(d, 4 * d)
        self.act = GELU()
        self.fc2 = Linear(4 * d, d)
        self.drop = Dropout(dropout)

    def forward(self, x):
        a = self.fc1.forward(x)
        b = self.act.forward(a)
        c = self.fc2.forward(b)
        return self.drop(c)


# Causal self attention. Each position can only attend to itself and earlier positions.
# Shapes use B = batch, T = sequence length, H = heads, hs = head_size.
class MultiHeadAttention(Module):
    def __init__(self, d, n_heads, block_size, dropout=0.0):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(
                "d and n_heads are not compatible. Make sure d is a multiple of n_heads."
            )
        self.d = d
        self.n_heads = n_heads
        self.block_size = block_size
        self.head_size = int(d / n_heads)
        self.q_map = Linear(in_features=d, out_features=d)
        self.k_map = Linear(in_features=d, out_features=d)
        self.v_map = Linear(in_features=d, out_features=d)
        self.out_proj = Linear(in_features=d, out_features=d)
        # 0 on and below the diagonal, -inf above it. Adding it to the scores
        # makes softmax give zero weight to future positions.
        self.mask = np.where(np.tri(block_size, block_size, dtype="bool"), 0.0, -np.inf)
        self.attn_drop = Dropout(dropout)
        self.resid_drop = Dropout(dropout)

    def forward(self, x):
        Q = self.q_map(x)
        K = self.k_map(x)
        V = self.v_map(x)

        # Split d into heads and move the head axis forward, (B, T, d) to (B, H, T, hs)
        Q_reshape = Q.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_size)
        K_reshape = K.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_size)
        V_reshape = V.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_size)

        Q_reshape_t = Q_reshape.transpose(axes=(0, 2, 1, 3))
        K_reshape_t = K_reshape.transpose(axes=(0, 2, 1, 3))
        V_reshape_t = V_reshape.transpose(axes=(0, 2, 1, 3))

        # Scaled dot product scores of shape (B, H, T, T)
        S = Q_reshape_t @ K_reshape_t.transpose(axes=(0, 1, 3, 2)) / np.sqrt(self.head_size)
        S_masked = S + self.mask[:x.shape[1], :x.shape[1]]

        A = S_masked.softmax()
        A_drop = self.attn_drop(A)
        O = A_drop @ V_reshape_t
        # Merge the heads back together, (B, H, T, hs) to (B, T, d)
        O_t = O.transpose(axes=(0, 2, 1, 3))
        O_cat = O_t.reshape(x.shape[0], x.shape[1], self.d)

        Y = self.out_proj(O_cat)
        return self.resid_drop(Y)


# One transformer block using pre norm and residual connections
class Block(Module):
    def __init__(self, d, n_heads, block_size, dropout=0.0):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.block_size = block_size
        self.ln1 = LayerNorm(d)
        self.attn = MultiHeadAttention(
            d=d, n_heads=n_heads, block_size=block_size, dropout=dropout
        )
        self.ln2 = LayerNorm(d)
        self.ff = FeedForward(d, dropout)

    def forward(self, x):
        A = x + self.attn(self.ln1(x))
        B = A + self.ff(self.ln2(A))

        return B


# Lookup table that maps integer ids to learned vectors
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Parameter.normal((num_embeddings, embedding_dim), std=0.02)

    def forward(self, ids):
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("ids must be of integer dtype.")
        return self.weight[ids]


# Decoder only transformer language model. Takes token ids of shape (B, T)
# and returns logits for the next token at every position, shape (B, T, vocab_size).
class GPT(Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size  # Maximum context length
        self.tok_emb = Embedding(self.vocab_size, self.d_model)
        self.pos_emb = Embedding(self.block_size, self.d_model)
        self.blocks = [
            Block(self.d_model, self.n_heads, self.block_size, dropout) for _ in range(n_layers)
        ]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, std=0.02)
        self.emb_drop = Dropout(dropout)

    def forward(self, ids):
        if ids.ndim != 2:
            raise TypeError(f"ids has dimension {ids.ndim}, but dimension 2 is required.")
        elif not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("ids must be of integer dtype.")
        elif ids.shape[1] > self.block_size:
            raise ValueError("ids is too long for block_size.")

        n = ids.shape[1]

        tok = self.tok_emb(ids)
        pos = self.pos_emb(np.arange(n))
        x = self.emb_drop(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    # Autoregressive sampling. Appends one token at a time to ids, which has shape (1, T).
    def generate(self, ids, max_new_tokens, rng, temperature=1.0):
        if ids.ndim != 2:
            raise TypeError("ids must have dim 2.")
        if ids.shape[0] != 1:
            raise TypeError("ids can only have one row.")
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        rng = np.random.default_rng(rng)
        with NoGrad(), EvalMode():
            for _ in range(max_new_tokens):
                # Only the last block_size tokens fit into the model
                context = ids[:, -self.block_size:]
                logits = self(context)
                # Logits for the token that follows the last position
                z = logits.data[0, -1, :]

                c = sample_text(z, temperature=temperature, rng=rng)
                ids = np.append(ids, [[c]], axis=1)
        return ids


# Inverted dropout. Zeroes each value with probability p during training and
# scales the rest by 1 / (1 - p) so nothing needs to change at eval time.
class Dropout(Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0 or not _training_enabled:
            return x

        mask = get_rng().binomial(n=1, p=(1 - self.p), size=x.shape)
        scale = (mask / (1 - self.p)).astype(x.dtype)
        return x * scale


# Context manager that turns off dropout, used for evaluation and sampling
class EvalMode:
    def __init__(self):
        pass

    def __enter__(self):
        global _training_enabled
        self.prev_value = _training_enabled
        _training_enabled = False

    def __exit__(self, exc_type, exc, tb):
        global _training_enabled
        _training_enabled = self.prev_value


def one_hot_encoding(array, n_labels):
    encoded_labels = np.zeros((array.size, n_labels), dtype=int)
    encoded_labels[np.arange(array.size), array] = 1
    return encoded_labels


# Mean cross entropy between logits of shape (..., n_labels) and integer class
# targets of shape (...). Works for both MNIST batches and GPT sequences.
def cross_entropy(logits, targets):
    n_labels = logits.shape[-1]
    if isinstance(targets, Tensor):
        targets = targets.data

    if logits.shape[:-1] != targets.shape:
        raise ValueError("Array and Labels don't have compatible shapes.")
    elif not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("Labels must be of integer dtype.")
    elif targets.max() >= n_labels or targets.min() < 0:
        raise ValueError("Labels must all be valid indices.")

    # Flatten everything into one batch of predictions
    reindexed_targets = targets.reshape(-1)
    reindexed_logits = logits.reshape(-1, logits.shape[-1])
    N = reindexed_logits.shape[0]

    # Pick the log probability of the correct class for every row
    return -1 * (reindexed_logits.log_softmax()[np.arange(N), reindexed_targets]).mean()


# Draws one token id from the logits z. Lower temperature makes the choice
# more greedy, higher temperature makes it more random.
def sample_text(z, temperature, rng=None):
    rng = np.random.default_rng(rng)
    z = z.astype(np.float64)
    p = np.exp((z - z.max()) / temperature) / np.exp((z - z.max()) / temperature).sum()
    return rng.choice(len(z), p=p)
