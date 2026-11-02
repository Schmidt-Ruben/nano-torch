import numpy as np

class CharTokenizer:
    def __init__(self, text):
        if not str:
            raise ValueError("Text can't be empty.")
        
        self.text = text
        self.chars = sorted(set(text))
        stoi = dict()
        for char, idx in enumerate(self.chars):
            stoi[char] = idx
        self.stoi = {char: idx for idx, char in enumerate(self.chars)}
        self.itos = {v: k for k, v in self.stoi.items()}
        self.V = len(self.chars)

    def encode(self, s):
        unknown_characters = [char for char in enumerate(s) if char[1] not in self.stoi.keys()]
        if unknown_characters:
            raise ValueError(f"Unknown characters at position: {unknown_characters}")
        ids = np.array([self.stoi[char] for char in s], dtype='int64')
        return ids

    def decode(self, ids):
        if ids.ndim != 1:
            raise ValueError('ids musst be of dimension 1.')
        invalid_ids = [id for id in ids if id < 0 or id >= self.V]
        if invalid_ids:
            raise ValueError(f'Invalid ids: {invalid_ids}')
        s = "".join([self.itos[id] for id in ids])
        return s
        

def one_hot_encoding(array, labels):
    encoded_labels = np.zeros((array.size, labels), dtype=int)
    encoded_labels[np.arange(array.size), array] = 1
    return encoded_labels


def get_batches(X, Y, batch_size, shuffle=True):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for start in range(0, len(X), batch_size):
        batch = idx[start:start + batch_size]
        yield X[batch], Y[batch]


def load_text(path):
    with open(file=path, mode='r', encoding='utf-8') as file:
        str = file.read()
    return str

def train_val_split(data, train_frac=0.9):
    if data.ndim != 1:
        raise ValueError("Data musst be of dimension 1.")
    if train_frac > 1 or train_frac < 0:
        raise ValueError("Train Eval split musst be between 0 and 1.")
    n = np.floor(len(data) * train_frac)
    return (data[:n], data[n:])

def get_text_batch(data, batch_size, block_size, rng=0):
    if data.ndim != 1:
        raise ValueError("Data musst be of dimension 1.")
    rng = np.random.default_rng(seed=rng)
    start_ids = rng.integers(low=0, high=(data.size - block_size), size=(batch_size, 1))
    index_matrix = start_ids + np.arange(0, block_size-1)
    x_batch = data[index_matrix]
    y_batch = data[index_matrix + 1]

    return x_batch, y_batch
    