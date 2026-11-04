# Trains a one hidden layer MLP on MNIST with the framework and reports test accuracy
# after every epoch

from data import get_batches, load_mnist_images, load_mnist_labels
from nn import Sequential, Linear, ReLU, cross_entropy
from optim import SGD
from tensor import Tensor, NoGrad

EPOCHS = 5
BATCH_SIZE = 128


X_train = load_mnist_images("mnist_dataset/train-images.idx3-ubyte")
Y_train = load_mnist_labels("mnist_dataset/train-labels.idx1-ubyte")
X_test = load_mnist_images("mnist_dataset/t10k-images.idx3-ubyte")
Y_test = load_mnist_labels("mnist_dataset/t10k-labels.idx1-ubyte")

model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
)
optimizer = SGD(model.parameters(), lr=0.1)

for epoch in range(EPOCHS):
    total_loss = 0.0
    n_batches = 0

    for X_batch, Y_batch in get_batches(X_train, Y_train, batch_size=BATCH_SIZE):
        x = Tensor(X_batch)
        y = Tensor(Y_batch, dtype="int64")

        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    # Evaluate on the test set without building a graph
    with NoGrad():
        test_logits = model(Tensor(X_test))
        predictions = test_logits.data.argmax(axis=1)
        accuracy = (predictions == Y_test).mean()
        print(
            f"epoch {epoch + 1}: train loss {total_loss / n_batches:.4f}, "
            f"test accuracy {accuracy:.4f}"
        )
