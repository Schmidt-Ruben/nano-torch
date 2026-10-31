import numpy as np

from manual_mnist import ROOT, forward, X_test, Y_test

loaded_model = np.load(ROOT / "model_checkpoints" / "model_final_weights.npz")

W1 = loaded_model["W1"]
b1 = loaded_model["b1"]
W2 = loaded_model["W2"]
b2 = loaded_model["b2"]

predictions = np.argmax(forward(X_test, W1, b1, W2, b2)[3], axis=1)
matches = np.where(predictions == Y_test, 1, 0)

print(f"Accuracy on unseen Data: {np.average(matches) * 100}%")

# print(predictions[:20], Y_test[:20], matches[:20],)
