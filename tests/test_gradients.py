# Checks every backward pass against a numerical gradient.
#
# For each input element x_i the numerical gradient is the central difference
#     (f(x_i + eps) - f(x_i - eps)) / (2 * eps)
# which needs no knowledge of the backward pass, so it is an independent reference.

import numpy as np
import pytest

from nn import (
    GELU, GPT, Block, Embedding, FeedForward, LayerNorm, Linear, MultiHeadAttention,
    cross_entropy,
)
from tensor import NoGrad, Tensor

EPS = 1e-6
RNG = np.random.default_rng(0)


def randn(*shape):
    return RNG.normal(size=shape)


def positive(*shape):  # For log, division and fractional powers
    return RNG.uniform(0.5, 2.0, size=shape)


def away_from_zero(*shape):  # ReLU has a kink at 0 where finite differences fail
    x = randn(*shape)
    return x + 0.2 * np.sign(x)


# f() must return a scalar Tensor computed from tensors. Compares the gradient that
# backward writes into every tensor with the central difference.
def gradcheck(f, tensors, rtol=1e-5, atol=1e-8):
    for t in tensors:
        t.grad = np.zeros_like(t.data)
    f().backward()

    for t in tensors:
        numeric = np.zeros_like(t.data)
        for i in np.ndindex(t.shape):
            original = t.data[i]
            t.data[i] = original + EPS
            with NoGrad():
                plus = f().item()
            t.data[i] = original - EPS
            with NoGrad():
                minus = f().item()
            t.data[i] = original
            numeric[i] = (plus - minus) / (2 * EPS)

        np.testing.assert_allclose(t.grad, numeric, rtol=rtol, atol=atol)


# Reduces a non-scalar output to a scalar with fixed random weights. A plain sum would
# give every output element the same upstream gradient and could hide mistakes.
def check_output(fn, tensors):
    weights = RNG.normal(size=fn().shape)
    gradcheck(lambda: (fn() * weights).sum(), tensors)


W_FIXED = randn(5, 3)

OP_CASES = [
    ("add", lambda a, b: a + b, [randn(3, 4), randn(3, 4)]),
    ("add broadcast row", lambda a, b: a + b, [randn(3, 4), randn(4)]),
    ("add broadcast column", lambda a, b: a + b, [randn(3, 4), randn(3, 1)]),
    ("sub broadcast leading axis", lambda a, b: a - b, [randn(2, 3, 4), randn(3, 4)]),
    ("mul broadcast", lambda a, b: a * b, [randn(3, 4), randn(1, 4)]),
    ("truediv", lambda a, b: a / b, [randn(3, 4), positive(3, 4)]),
    ("truediv broadcast", lambda a, b: a / b, [randn(3, 4), positive(4)]),
    ("neg", lambda a: -a, [randn(3, 4)]),
    ("pow 3", lambda a: a**3, [randn(3, 4)]),
    ("pow 0.5", lambda a: a**0.5, [positive(3, 4)]),
    ("number ops", lambda a: 2.0 - a * 3.0 + 1.0 / a, [positive(3, 4)]),
    ("matmul", lambda a, b: a @ b, [randn(3, 4), randn(4, 5)]),
    ("batched matmul", lambda a, b: a @ b, [randn(2, 3, 4), randn(2, 4, 5)]),
    ("broadcast matmul", lambda a, b: a @ b, [randn(2, 3, 4), randn(4, 5)]),
    ("4d matmul", lambda a, b: a @ b, [randn(2, 2, 3, 4), randn(2, 2, 4, 3)]),
    ("ndarray @ tensor", lambda a: W_FIXED @ a, [randn(3, 4)]),
    ("exp", lambda a: a.exp(), [randn(3, 4)]),
    ("log", lambda a: a.log(), [positive(3, 4)]),
    ("relu", lambda a: a.relu(), [away_from_zero(3, 4)]),
    ("tanh", lambda a: a.tanh(), [randn(3, 4)]),
    ("sum all", lambda a: a.sum(), [randn(3, 4)]),
    ("sum axis", lambda a: a.sum(axis=0), [randn(3, 4)]),
    ("sum keepdims", lambda a: a.sum(axis=-1, keepdims=True), [randn(3, 4)]),
    ("sum tuple axis", lambda a: a.sum(axis=(0, 2)), [randn(2, 3, 4)]),
    ("mean all", lambda a: a.mean(), [randn(3, 4)]),
    ("mean axis", lambda a: a.mean(axis=1), [randn(3, 4)]),
    ("mean keepdims", lambda a: a.mean(axis=-1, keepdims=True), [randn(2, 3, 4)]),
    ("reshape", lambda a: a.reshape(4, 3), [randn(3, 4)]),
    ("transpose", lambda a: a.transpose(), [randn(3, 4)]),
    ("transpose axes", lambda a: a.transpose(axes=(0, 2, 1, 3)), [randn(2, 3, 4, 2)]),
    ("getitem slice", lambda a: a[1:, :2], [randn(3, 4)]),
    ("getitem repeated ids", lambda a: a[np.array([0, 2, 0, 0])], [randn(3, 4)]),
    ("getitem pairs", lambda a: a[np.arange(3), np.array([1, 0, 1])], [randn(3, 4)]),
    ("softmax", lambda a: a.softmax(), [randn(3, 4)]),
    ("softmax axis 0", lambda a: a.softmax(axis=0), [randn(3, 4)]),
    ("log_softmax", lambda a: a.log_softmax(), [randn(3, 4)]),
    ("reused node", lambda a: a * a + a.exp() * a, [randn(3, 4)]),
]


@pytest.mark.parametrize("fn, inputs", [c[1:] for c in OP_CASES], ids=[c[0] for c in OP_CASES])
def test_op_gradients(fn, inputs):
    tensors = [Tensor(x, requires_grad=True) for x in inputs]
    check_output(lambda: fn(*tensors), tensors)


def test_linear():
    layer = Linear(3, 4)
    x = Tensor(randn(5, 3), requires_grad=True)
    check_output(lambda: layer(x), [x] + layer.parameters())


def test_linear_without_bias():
    layer = Linear(3, 4, bias=False)
    x = Tensor(randn(5, 3), requires_grad=True)
    check_output(lambda: layer(x), [x] + layer.parameters())


def test_layer_norm():
    layer = LayerNorm(6)
    # Non-default scale and shift so their gradients are not trivial
    layer.gamma.data = randn(6)
    layer.beta.data = randn(6)
    x = Tensor(randn(2, 3, 6), requires_grad=True)
    check_output(lambda: layer(x), [x] + layer.parameters())


def test_gelu():
    x = Tensor(randn(3, 4), requires_grad=True)
    check_output(lambda: GELU()(x), [x])


def test_feed_forward():
    layer = FeedForward(4)
    x = Tensor(randn(2, 3, 4), requires_grad=True)
    check_output(lambda: layer(x), [x] + layer.parameters())


def test_multi_head_attention():
    # T = 4 is shorter than block_size = 5, so the mask is sliced as in sampling
    layer = MultiHeadAttention(d=8, n_heads=2, block_size=5)
    x = Tensor(randn(2, 4, 8), requires_grad=True)
    check_output(lambda: layer(x), [x] + layer.parameters())


def test_block():
    layer = Block(d=8, n_heads=2, block_size=5)
    x = Tensor(randn(2, 4, 8), requires_grad=True)
    check_output(lambda: layer(x), [x] + layer.parameters())


def test_embedding_with_repeated_ids():
    layer = Embedding(5, 3)
    ids = np.array([[0, 2, 2], [4, 0, 2]])
    check_output(lambda: layer(ids), layer.parameters())


@pytest.mark.parametrize("shape", [(4, 5), (2, 3, 5)], ids=["2d", "3d"])
def test_cross_entropy(shape):
    logits = Tensor(randn(*shape), requires_grad=True)
    targets = RNG.integers(0, shape[-1], size=shape[:-1])
    gradcheck(lambda: cross_entropy(logits, targets), [logits])


def test_gpt_end_to_end():
    model = GPT(vocab_size=7, d_model=8, n_heads=2, n_layers=2, block_size=5)
    ids = RNG.integers(0, 7, size=(2, 4))
    targets = RNG.integers(0, 7, size=(2, 4))
    gradcheck(lambda: cross_entropy(model(ids), targets), model.parameters())
