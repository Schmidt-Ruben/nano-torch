import numpy as np


def one_hot_encoding(array, labels):
    encoded_labels = np.zeros((array.size, labels), dtype=int)
    encoded_labels[np.arange(array.size), array] = 1
    return encoded_labels


def get_batches(X, Y, batch_size, shuffle=True):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for start in range(0, len(X), batch_size):
        batch = idx[start:start + batch_size]
        yield X[batch], Y[batch]
