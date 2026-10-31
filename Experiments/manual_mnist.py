import struct
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import get_batches  

BATCH_SIZE = 128
LEARNING_RATE = 0.1
EPOCHS = 5
HIDDEN_LAYER_NEURONS = 256
CHECKPOINT_EVERY = 10  # epochs between intermediate checkpoints

DATASET_DIR = ROOT / "mnist_dataset"
CHECKPOINT_DIR = ROOT / "model_checkpoints"


def load_mnist_images(filename):
    with open(filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))

        # Load the actual data
        images = np.fromfile(f, dtype=np.uint8)

        # Flatten to 1 x 784 and normalise values to [0, 1]
        images = images.reshape(num_images, rows * cols).astype(np.float32)
        images /= 255.0

        return images


def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))

        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def one_hot_encoding(array):
    encoded_labels = np.zeros((array.size, 10), dtype=int)
    encoded_labels[np.arange(array.size), array] = 1
    return encoded_labels


def softmax(z):
    soft = np.exp(z)
    return np.divide(soft, np.sum(soft, axis=1, keepdims=True))


def ReLu(z):
    return np.maximum(z, 0)


def cross_entropy_loss(output, label):
    # 1e-8 prevents log(0) spikes
    return -np.sum(label * np.log(output + 1e-8)) / label.shape[0]


def forward(x, W1, b1, W2, b2):
    Z1 = x @ W1 + b1  # first layer
    A1 = ReLu(Z1)
    Z2 = A1 @ W2 + b2  # second layer
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def backward(x, labels, Z1, A1, Z2, A2, W1, b1, W2, b2):
    dZ2 = (1 / labels.shape[0]) * (A2 - labels)
    dW2 = np.transpose(A1) @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ np.transpose(W2)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.transpose(x) @ dZ1
    db1 = np.sum(dZ1, axis=0)

    # Update hidden layer
    W1 -= dW1 * LEARNING_RATE
    b1 -= db1 * LEARNING_RATE
    # Update output layer
    W2 -= dW2 * LEARNING_RATE
    b2 -= db2 * LEARNING_RATE

    return W1, b1, W2, b2


X_train = load_mnist_images(DATASET_DIR / "train-images.idx3-ubyte")
Y_train = load_mnist_labels(DATASET_DIR / "train-labels.idx1-ubyte")
X_test = load_mnist_images(DATASET_DIR / "t10k-images.idx3-ubyte")
Y_test = load_mnist_labels(DATASET_DIR / "t10k-labels.idx1-ubyte")

Y_train_encoded = one_hot_encoding(Y_train)


def train():
    # Initialise weights (He initialisation, variance 2 / N_in) and biases
    W1 = np.random.normal(loc=0.0, scale=np.sqrt(2 / 784), size=(784, HIDDEN_LAYER_NEURONS))
    b1 = np.zeros(shape=(1, HIDDEN_LAYER_NEURONS))

    W2 = np.random.normal(
        loc=0.0, scale=np.sqrt(2 / HIDDEN_LAYER_NEURONS), size=(HIDDEN_LAYER_NEURONS, 10)
    )
    b2 = np.zeros(shape=(1, 10))

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    losses = np.array([])

    for epoch in range(EPOCHS):
        batch_losses = np.array([])

        for X_batch, Y_batch in get_batches(X_train, Y_train_encoded, batch_size=BATCH_SIZE):
            x = X_batch
            labels = Y_batch

            Z1, A1, Z2, A2 = forward(x, W1, b1, W2, b2)
            batch_losses = np.append(batch_losses, cross_entropy_loss(A2, labels))
            W1, b1, W2, b2 = backward(x, labels, Z1, A1, Z2, A2, W1, b1, W2, b2)

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            filename = CHECKPOINT_DIR / f"model_checkpoint_epoch_{epoch + 1}.npz"
            np.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2)
            print(f"Saved intermediate weights to {filename}")

        losses = np.concatenate((losses, batch_losses))

    np.savez(CHECKPOINT_DIR / "model_final_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    print("Training complete. Final weights saved.")
    # plt.plot(losses)
    # plt.show()


if __name__ == "__main__":
    train()
