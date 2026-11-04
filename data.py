import struct

import numpy as np


# Character level tokenizer. Every distinct character in the text gets one id.
class CharTokenizer:
    def __init__(self, text):
        if not text:
            raise ValueError("Text can't be empty.")

        self.text = text
        self.chars = sorted(set(text))  # Sorted so the ids are the same on every run
        self.stoi = {char: idx for idx, char in enumerate(self.chars)}  # string to id
        self.itos = {v: k for k, v in self.stoi.items()}  # id to string
        self.V = len(self.chars)  # Vocabulary size

    def encode(self, s):
        unknown_characters = [char for char in enumerate(s) if char[1] not in self.stoi.keys()]
        if unknown_characters:
            raise ValueError(f"Unknown characters at position: {unknown_characters}")
        ids = np.array([self.stoi[char] for char in s], dtype="int64")
        return ids

    def decode(self, ids):
        ids = np.asarray(ids)
        if ids.ndim != 1:
            raise ValueError("ids must be of dimension 1.")
        invalid_ids = [i for i in ids if i < 0 or i >= self.V]
        if invalid_ids:
            raise ValueError(f"Invalid ids: {invalid_ids}")
        s = "".join([self.itos[i] for i in ids])
        return s


# Reads MNIST images in the IDX format as float32 rows of 784 pixels in [0, 1]
def load_mnist_images(filename):
    with open(filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        # Load the actual data
        images = np.fromfile(f, dtype=np.uint8)

        # Flatten to 1 x 784 and normalise values to [0, 1]
        images = images.reshape(num_images, rows * cols).astype(np.float32)
        images /= 255.0

        return images


# Reads MNIST labels in the IDX format as uint8 digits 0 to 9
def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))

        labels = np.fromfile(f, dtype=np.uint8)
        return labels


# Yields (X, Y) mini batches that cover the whole dataset once
def get_batches(X, Y, batch_size, shuffle=True):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for start in range(0, len(X), batch_size):
        batch = idx[start:start + batch_size]
        yield X[batch], Y[batch]


def load_text(path):
    with open(file=path, mode="r", encoding="utf-8") as file:
        text = file.read()
    return text


# Splits without shuffling so the validation set is a separate part of the text
def train_val_split(data, train_frac=0.9):
    if data.ndim != 1:
        raise ValueError("Data must be of dimension 1.")
    if train_frac > 1 or train_frac < 0:
        raise ValueError("Train Eval split must be between 0 and 1.")
    n = int(len(data) * train_frac)
    return (data[:n], data[n:])


# Samples batch_size random windows of length block_size from data.
# y is x shifted one step to the right, so y[i, t] is the token that follows x[i, t].
def get_text_batch(data, batch_size, block_size, rng=None):  # rng is np random generator
    rng = np.random.default_rng(rng)
    if data.ndim != 1:
        raise ValueError("Data must be of dimension 1.")
    start_ids = rng.integers(low=0, high=(data.size - block_size), size=(batch_size, 1))
    # Each row holds the indices start, start + 1, ..., start + block_size - 1
    index_matrix = start_ids + np.arange(0, block_size)
    x_batch = data[index_matrix]
    y_batch = data[index_matrix + 1]

    return x_batch, y_batch
