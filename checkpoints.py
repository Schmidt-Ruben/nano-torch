import numpy as np

from data import CharTokenizer
from nn import GPT


# Rebuilds a GPT and its tokenizer from a file written by save_gpt_weights
def load_gpt_weights(path):
    with np.load(path) as ckpt:
        model = GPT(
            int(ckpt["vocab_size"]),
            int(ckpt["d_model"]),
            int(ckpt["n_heads"]),
            int(ckpt["n_layers"]),
            int(ckpt["block_size"]),
            float(ckpt["dropout"]),
        )
        params = model.parameters()
        assert int(ckpt["n_params"]) == len(params), "parameter count mismatch"
        # Parameters are matched by position, so the model must list them in the saved order
        for i, p in enumerate(params):
            assert p.data.shape == ckpt[f"p{i}"].shape, f"shape mismatch at p{i}"
            p.data[...] = ckpt[f"p{i}"]
        tok = CharTokenizer(str(ckpt["chars"]))
    return model, tok


# Saves the model config, the vocabulary and every parameter as p0, p1, ... in one .npz file
def save_gpt_weights(path, model, tok):
    params = {f"p{i}": p.data for i, p in enumerate(model.parameters())}
    config = {
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "n_layers": model.n_layers,
        "block_size": model.block_size,
        "dropout": model.emb_drop.p,
    }
    np.savez(path, chars=np.array("".join(tok.chars)), n_params=len(params), **config, **params)
