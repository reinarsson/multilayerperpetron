"""Train an MLP on MNIST handwritten digit images.

Each sample is a 28x28 grayscale image (784 float32 values in [0.0, 1.0])
representing a single handwritten digit (0-9).

The network takes a flattened 784-input vector and maps it to a
10-class one-hot output via a single hidden layer.

Note: this implementation uses pure Python backprop and is intentionally
trained on a small subset (1000 samples). For full MNIST training,
use a framework such as PyTorch or TensorFlow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multilayer_perceptron import MultiLayerPerceptron

INPUT_UNITS = 784
HIDDEN_UNITS = 64
OUTPUT_UNITS = 10
EPOCHS = 10
ETA = 0.1
TRAIN_SAMPLES = 1000
TEST_SAMPLES = 200


def load_mnist(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download MNIST and return normalised float32 arrays.

    Returns:
        X_train: (n_train, 784) float32 array, values in [0.0, 1.0]
        X_test:  (n_test,  784) float32 array, values in [0.0, 1.0]
        y_train: (n_train,) int array of labels 0-9
        y_test:  (n_test,)  int array of labels 0-9
    """
    print("Loading MNIST (this may take a moment on first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")

    X = mnist.data.astype(np.float32) / 255.0   # normalise to [0.0, 1.0]
    y = mnist.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=n_train, test_size=n_test, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def to_one_hot(label: int, n_classes: int = OUTPUT_UNITS) -> list[float]:
    """Convert an integer label to a one-hot vector."""
    vec = [0.0] * n_classes
    vec[label] = 1.0
    return vec


def predict(mlp: MultiLayerPerceptron, x: np.ndarray) -> int:
    """Return the predicted digit for a single 784-element input vector."""
    output = mlp.run(x.tolist())
    return int(np.argmax(output))


def train(mlp: MultiLayerPerceptron, X: np.ndarray, y: np.ndarray, epochs: int) -> list[float]:
    """Train the MLP and return MSE per epoch."""
    mse_history = []
    for epoch in range(epochs):
        mse = 0.0
        for image, label in zip(X, y):
            mse += mlp.bp(image.tolist(), to_one_hot(label))
        mse /= len(X)
        mse_history.append(round(float(mse), 6))
        print(f"Epoch {epoch + 1}/{epochs}  MSE: {mse:.4f}")
    return mse_history


def evaluate(mlp: MultiLayerPerceptron, X: np.ndarray, y: np.ndarray) -> float:
    """Return the accuracy of the MLP on the provided dataset."""
    correct = sum(predict(mlp, image) == label for image, label in zip(X, y))
    return correct / len(y)


def show_sample(image: np.ndarray, label: int, predicted: int) -> None:
    """Print an ASCII preview of a single 28x28 image with its labels."""
    grid = image.reshape(28, 28)
    for row in grid:
        print("".join("#" if v > 0.5 else "." for v in row))
    print(f"True: {label}  Predicted: {predicted}\n")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_mnist(TRAIN_SAMPLES, TEST_SAMPLES)

    print(f"\nSample shape: {X_train[0].reshape(28, 28).shape}  "
          f"dtype: {X_train[0].dtype}  "
          f"range: [{X_train[0].min():.2f}, {X_train[0].max():.2f}]")
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.\n")

    mlp = MultiLayerPerceptron(
        layers=[INPUT_UNITS, HIDDEN_UNITS, OUTPUT_UNITS],
        eta=ETA,
    )

    mse_history = train(mlp, X_train, y_train, EPOCHS)

    train_accuracy = evaluate(mlp, X_train, y_train)
    test_accuracy = evaluate(mlp, X_test, y_test)
    print(f"\nTrain accuracy: {train_accuracy * 100:.1f}%")
    print(f"Test accuracy:  {test_accuracy * 100:.1f}%")

    print("\nSample predictions:")
    for image, label in zip(X_test[:3], y_test[:3]):
        show_sample(image, label, predict(mlp, image))

    output_path = Path(__file__).parent.parent / "output" / "mnist.npz"
    mlp.save(
        output_path,
        metadata={
            "training": {
                "epochs": EPOCHS,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "mse_initial": mse_history[0],
                "mse_final": mse_history[-1],
                "mse_per_epoch": mse_history,
            },
            "accuracy": {
                "train": round(train_accuracy, 4),
                "test": round(test_accuracy, 4),
            },
        },
    )
    print(f"Model saved to {output_path}")
