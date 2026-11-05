# Behaviour of the framework apart from gradient values: dtypes, graph switches,
# optimisers, the GPT, checkpoints and the data helpers.

import numpy as np
import pytest

import tensor
from checkpoints import load_gpt_weights, save_gpt_weights
from data import CharTokenizer, get_text_batch, train_val_split
from nn import GPT, Dropout, EvalMode, Linear, cross_entropy
from optim import SGD, Adam, clip_grad_norm
from tensor import NoGrad, Tensor


# nn and optim are imported above, before the dtype is changed. All arrays they
# create must still follow tensor.DTYPE at the time the tensors are made.
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dtype_set_after_import_is_used_everywhere(monkeypatch, dtype):
    monkeypatch.setattr(tensor, "DTYPE", dtype)
    layer = Linear(3, 2)
    x = Tensor(np.ones((4, 3)))

    loss = Dropout(0.5)(layer(x)).sum()
    loss.backward()
    assert loss.dtype == dtype
    assert loss.grad.dtype == dtype

    optimizer = Adam(layer.parameters())
    optimizer.zero_grad()
    assert all(p.grad.dtype == dtype for p in layer.parameters())
    assert all(m.dtype == dtype for m in optimizer.m + optimizer.v)


def test_no_grad_builds_no_graph():
    x = Tensor(np.ones(3), requires_grad=True)
    with NoGrad():
        y = (x * 2).exp().sum()
    assert y._prev == ()
    assert not y.requires_grad


def test_dropout_is_off_in_eval_mode():
    x = Tensor(np.ones((10, 10)))
    with EvalMode():
        assert np.array_equal(Dropout(0.5)(x).data, x.data)


def test_dropout_zeroes_and_rescales_in_training():
    out = Dropout(0.5)(Tensor(np.ones((200, 200)))).data
    # Kept values are scaled by 1 / (1 - p) so the expected value stays 1
    assert set(np.unique(out)) == {0.0, 2.0}
    assert abs(out.mean() - 1.0) < 0.02


def test_parameters_are_found_in_nested_modules_and_lists():
    model = GPT(vocab_size=7, d_model=8, n_heads=2, n_layers=2, block_size=5)
    params = model.parameters()
    # 2 embeddings + 2 blocks * (ln1 2 + attention 8 + ln2 2 + feed forward 4) + ln_f 2 + head 2
    assert len(params) == 2 + 2 * 16 + 2 + 2
    assert len({id(p) for p in params}) == len(params)


def test_gpt_is_causal():
    model = GPT(vocab_size=7, d_model=8, n_heads=2, n_layers=2, block_size=6)
    ids = np.array([[1, 2, 3, 4, 5, 6]])
    changed = ids.copy()
    changed[0, 3] = 0
    with NoGrad():
        before = model(ids).data
        after = model(changed).data
    # Changing token 3 may only affect the logits from position 3 onwards
    np.testing.assert_array_equal(before[:, :3], after[:, :3])
    assert not np.allclose(before[:, 3:], after[:, 3:])


def test_gpt_generate():
    model = GPT(vocab_size=7, d_model=8, n_heads=2, n_layers=1, block_size=4)
    out = model.generate(np.array([[1, 2]]), max_new_tokens=10, rng=0)
    assert out.shape == (1, 12)
    assert out.min() >= 0 and out.max() < 7


def test_gpt_overfits_a_single_batch():
    model = GPT(vocab_size=7, d_model=16, n_heads=2, n_layers=1, block_size=8)
    optimizer = Adam(model.parameters(), lr=1e-2)
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 7, size=(2, 8)), rng.integers(0, 7, size=(2, 8))

    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        loss = cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1 * losses[0]


def test_cross_entropy_value():
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    targets = np.array([0, 2])
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    expected = -np.log(probs[np.arange(2), targets]).mean()
    assert cross_entropy(Tensor(logits), targets).item() == pytest.approx(expected)


def test_cross_entropy_rejects_invalid_targets():
    logits = Tensor(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        cross_entropy(logits, np.array([0, 3]))
    with pytest.raises(TypeError):
        cross_entropy(logits, np.array([0.0, 1.0]))


def test_sgd_step():
    layer = Linear(2, 2)
    p = layer.W
    p.grad = np.full(p.shape, 0.5)
    before = p.data.copy()
    SGD([p], lr=0.1).step()
    np.testing.assert_allclose(p.data, before - 0.05)


def test_adam_first_step_moves_by_learning_rate():
    # After bias correction the first update is lr * g / abs(g), so every weight moves by lr
    layer = Linear(2, 2)
    p = layer.W
    p.grad = np.array([[0.3, -2.0], [1e-3, -0.7]])
    before = p.data.copy()
    Adam([p], lr=0.01).step()
    np.testing.assert_allclose(p.data, before - 0.01 * np.sign(p.grad), rtol=1e-4)


def test_clip_grad_norm():
    params = Linear(3, 3).parameters()
    for p in params:
        p.grad = np.full(p.shape, 2.0)
    norm = clip_grad_norm(params, max_norm=1.0)
    assert norm == pytest.approx(np.sqrt(12 * 4.0))
    clipped = np.sqrt(sum(np.sum(p.grad**2) for p in params))
    assert clipped == pytest.approx(1.0, rel=1e-5)

    # Gradients below the limit are left unchanged
    for p in params:
        p.grad = np.full(p.shape, 0.01)
    clip_grad_norm(params, max_norm=1.0)
    assert all(np.all(p.grad == 0.01) for p in params)


def test_checkpoint_round_trip(tmp_path):
    tok = CharTokenizer("hello world")
    model = GPT(vocab_size=tok.V, d_model=8, n_heads=2, n_layers=1, block_size=4, dropout=0.1)
    path = tmp_path / "gpt.npz"
    save_gpt_weights(path, model, tok)

    loaded, loaded_tok = load_gpt_weights(path)
    ids = tok.encode("hell").reshape(1, -1)
    with NoGrad(), EvalMode():
        np.testing.assert_array_equal(model(ids).data, loaded(ids).data)
    assert loaded_tok.chars == tok.chars
    assert loaded.emb_drop.p == 0.1


def test_tokenizer_round_trip():
    tok = CharTokenizer("To be, or not to be")
    assert tok.decode(tok.encode("be or not")) == "be or not"
    with pytest.raises(ValueError):
        tok.encode("z")


def test_text_batch_targets_are_shifted_inputs():
    data = np.arange(100)
    x, y = get_text_batch(data, batch_size=4, block_size=8, rng=0)
    assert x.shape == y.shape == (4, 8)
    np.testing.assert_array_equal(y, x + 1)


def test_train_val_split_keeps_order():
    train, val = train_val_split(np.arange(10), train_frac=0.8)
    np.testing.assert_array_equal(train, np.arange(8))
    np.testing.assert_array_equal(val, [8, 9])
