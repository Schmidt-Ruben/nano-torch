# Verification of the framework against the hand-written Manual MNIST
#
# Both start from the same weights and see the same batches. The script compares
# the loss, all four gradients and the weights after each SGD step.
#

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tensor

tensor.DTYPE = "float64"  # must be set before any Tensor is created

from tensor import Tensor  
from nn import Sequential, Linear, ReLU, cross_entropy  
from optim import SGD  

BATCH_SIZE = 128
HIDDEN = 256  # same as Manual
LEARNING_RATE = 0.1
STEPS = 3  # several steps also test zero_grad
TOLERANCE = 1e-8  # max relative error allowed

DATASET_DIR = ROOT / "mnist_dataset"


# Data loaders unchanged from Manual MNIST

def load_mnist_images(filename):
    with open(filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8)
        images = images.reshape(num_images, rows * cols).astype(np.float32)
        images /= 255.0
        return images


def load_mnist_labels(filename):
    with open(filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        return np.fromfile(f, dtype=np.uint8)


def one_hot(labels, n_classes=10):
    encoded = np.zeros((labels.size, n_classes))
    encoded[np.arange(labels.size), labels] = 1
    return encoded


# Manual MNIST  forward and backward math without the weight update

def manual_softmax(z):
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def manual_loss_and_grads(X, Y, W1, b1, W2, b2):
    N = X.shape[0]

    # forward
    Z1 = X @ W1 + b1
    A1 = np.maximum(Z1, 0)
    Z2 = A1 @ W2 + b2
    A2 = manual_softmax(Z2)
    loss = -np.sum(Y * np.log(A2 + 1e-8)) / N

    # backward
    dZ2 = (1 / N) * (A2 - Y)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}



results = []

#Helpers
def rel_error(a, b):
    # Largest difference relative to the size of the reference values
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return None
    scale = max(np.max(np.abs(b)), 1e-12)
    return np.max(np.abs(a - b)) / scale


def compare(name, ours, reference, tol=TOLERANCE):
    err = rel_error(ours, reference)
    if err is None:
        ok = False
        detail = (
            f"shape mismatch: framework {np.shape(ours)} vs manual {np.shape(reference)}"
        )
    else:
        ok, detail = err < tol, f"rel. error {err:.2e}"
    results.append(ok)
    print(f"    {'PASS' if ok else 'FAIL'}  {name:<16} {detail}")


# Setup for identical starting weights and batches for both

X_train = load_mnist_images(DATASET_DIR / "train-images.idx3-ubyte")
Y_train = one_hot(load_mnist_labels(DATASET_DIR / "train-labels.idx1-ubyte"))

rng = np.random.default_rng(42)
init = {
    "W1": rng.normal(0.0, np.sqrt(2 / 784), size=(784, HIDDEN)),
    "b1": np.zeros(HIDDEN),
    "W2": rng.normal(0.0, np.sqrt(2 / HIDDEN), size=(HIDDEN, 10)),
    "b2": np.zeros(10),
}

# Manual keeps its own copies of the weights
ref = {name: value.copy() for name, value in init.items()}

# Framework model with its weights overwritten by the same starting values
model = Sequential(Linear(784, HIDDEN), ReLU(), Linear(HIDDEN, 10))
params = {
    "W1": model.layers[0].W, "b1": model.layers[0].b,
    "W2": model.layers[2].W, "b2": model.layers[2].b,
}
for name, p in params.items():
    p.data = np.array(init[name], dtype=np.float64)
    p.grad = np.zeros_like(p.data)

optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

print(f"\nComparing framework and Manual: {STEPS} SGD steps, "
      f"batch {BATCH_SIZE}, hidden {HIDDEN}, lr {LEARNING_RATE}")


# Comparing several steps on fixed and consecutive batches

for step in range(STEPS):
    start = step * BATCH_SIZE
    X = X_train[start:start + BATCH_SIZE].astype(np.float64)
    Y = Y_train[start:start + BATCH_SIZE]
    print(f"\n  Step {step + 1}")

    # Manual MNIST
    ref_loss, ref_grads = manual_loss_and_grads(X, Y, **ref)

    # Framework
    optimizer.zero_grad()
    logits = model(Tensor(X))
    loss = cross_entropy(logits, Tensor(Y))
    loss.backward()

    # For Manual 1e-8 inside the log shifts its loss by about 1e-8, hence the looser tolerance
    compare("loss", loss.item(), ref_loss, tol=1e-6)
    for name in ("W1", "b1", "W2", "b2"):
        compare(f"grad {name}", params[name].grad, ref_grads[name])

    # Update both, then compare the new weights
    optimizer.step()
    for name in ref:
        ref[name] -= LEARNING_RATE * ref_grads[name]
    for name in ("W1", "b1", "W2", "b2"):
        compare(f"{name} after step", params[name].data, ref[name])


passed = sum(results)
print(f"\n{passed}/{len(results)} passed.")
if passed == len(results):
    print("The autograd reproduces Manual MNIST exactly.")
raise SystemExit(0 if passed == len(results) else 1)
