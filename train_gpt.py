# Trains a character level GPT on text_dataset/input.txt and saves checkpoints
# to gpt_checkpoints/
import numpy as np

from checkpoints import save_gpt_weights
from data import CharTokenizer, get_text_batch, load_text, train_val_split
from nn import GPT, EvalMode, cross_entropy
from optim import Adam, clip_grad_norm
from tensor import NoGrad, set_seed

TRAIN_SEED = 1234
EVAL_SEED = 2345
BLOCK_SIZE = 64  # Context length in characters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_STEPS = 5000
EVAL_INTERVAL = 50  # Steps between loss reports
EVAL_ITERS = 20  # Batches averaged per loss estimate
CHECKPOINT_INTERVAL = 500  # Steps between saved checkpoints
DROPOUT = 0.1


# Averages the loss over several random batches because a single batch is noisy
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, rng):
    results = dict()
    with NoGrad(), EvalMode():
        train_losses = []
        val_losses = []
        for _ in range(eval_iters):
            x_train, y_train = get_text_batch(train_data, batch_size, block_size, rng)
            logits_train = model(x_train)
            loss_train = cross_entropy(logits_train, y_train)
            train_losses.append(loss_train.item())

            x_val, y_val = get_text_batch(val_data, batch_size, block_size, rng)
            logits_val = model(x_val)
            loss_val = cross_entropy(logits_val, y_val)
            val_losses.append(loss_val.item())

        results["Train"] = np.mean(train_losses)
        results["Val"] = np.mean(val_losses)

    return results


# Sanity check that trains on one small fixed batch. If the model and the backward
# pass are correct the printed loss should drop close to zero.
def overfit_check(data):
    model = GPT(65, 128, 4, 4, 64)
    optimizer = Adam(model.parameters())

    steps = 300
    seed = 1234

    rng = np.random.default_rng(seed=seed)
    x, y = get_text_batch(data, 4, 64, rng)

    for step in range(steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            print(loss.item())


def train(model, optimizer, tok, train_data, val_data, rng_train, rng_eval):
    for step in range(MAX_STEPS):
        x, y = get_text_batch(train_data, BATCH_SIZE, BLOCK_SIZE, rng_train)
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        if step % EVAL_INTERVAL == 0:
            train_loss, val_loss = estimate_loss(
                model, train_data, val_data, BATCH_SIZE, BLOCK_SIZE, EVAL_ITERS, rng_eval
            ).values()
            print(f"Step:{step}, Train loss: {train_loss}, Evaluation loss: {val_loss}")
        if step % CHECKPOINT_INTERVAL == 0 and step != 0:
            save_gpt_weights(f"gpt_checkpoints/gpt_shakespeare_checkpoint{step}.npz", model, tok)


def main():
    # Seeds weight initialisation and dropout
    set_seed(TRAIN_SEED)
    # Separate generators so evaluation does not change which batches training sees
    rng_train = np.random.default_rng(seed=TRAIN_SEED)
    rng_eval = np.random.default_rng(seed=EVAL_SEED)

    text = load_text("text_dataset/input.txt")
    tok = CharTokenizer(text)
    data = tok.encode(text)
    train_data, val_data = train_val_split(data, train_frac=0.9)

    # d_model 128, 4 heads, 4 layers
    model = GPT(tok.V, 128, 4, 4, BLOCK_SIZE, DROPOUT)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, optimizer, tok, train_data, val_data, rng_train, rng_eval)
    save_gpt_weights("gpt_checkpoints/gpt_shakespeare.npz", model, tok)


if __name__ == "__main__":
    main()
