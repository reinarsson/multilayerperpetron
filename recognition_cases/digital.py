"""Train an MLP to recognise 7-segment display digits (0-9).

Each digit is encoded as a 7-bit input vector representing the segments:
    ─
   | |
    ─
   | |
    ─

The network maps 7-segment patterns to one-hot 10-class outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multilayer_perceptron import MultiLayerPerceptron

EPOCHS = 3000
HIDDEN_UNITS = 7
OUTPUT_UNITS = 10
INPUT_UNITS = 7
OUTPUT_PATH = Path(__file__).parent.parent / "output" / "digital.npz"

PATTERNS = [
    ([1, 1, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # 0
    ([0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),  # 1
    ([1, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),  # 2
    ([1, 1, 1, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # 3
    ([0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # 4
    ([1, 0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),  # 5
    ([1, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),  # 6
    ([1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),  # 7
    ([1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),  # 8
    ([1, 1, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),  # 9
]


def train(mlp: MultiLayerPerceptron, epochs: int) -> list[float]:
    """Train the network and return MSE per epoch."""
    mse_history = []
    for _ in range(epochs):
        mse = sum(mlp.bp(x, y) for x, y in PATTERNS) / len(PATTERNS)
        mse_history.append(round(float(mse), 6))
    print(f"Final MSE: {mse_history[-1]:.6f}")
    return mse_history


def evaluate(mlp: MultiLayerPerceptron) -> float:
    """Print output per digit and return accuracy (fraction of correct predictions)."""
    correct = 0
    for digit, (x, y) in enumerate(PATTERNS):
        output = mlp.run(x)
        predicted = int(output.argmax())
        expected = int(y.index(1))
        correct += predicted == expected
        print(f"{digit} = {output[0]:.10f}  predicted={predicted}")
    accuracy = correct / len(PATTERNS)
    print(f"Accuracy: {accuracy * 100:.1f}%")
    return accuracy


if __name__ == "__main__":
    mlp = MultiLayerPerceptron(layers=[INPUT_UNITS, HIDDEN_UNITS, OUTPUT_UNITS])
    mse_history = train(mlp, EPOCHS)
    accuracy = evaluate(mlp)
    mlp.save(
        OUTPUT_PATH,
        metadata={
            "training": {
                "epochs": EPOCHS,
                "samples": len(PATTERNS),
                "mse_initial": mse_history[0],
                "mse_final": mse_history[-1],
                "mse_per_epoch": mse_history,
            },
            "accuracy": {
                "train": accuracy,
            },
        },
    )
    print(f"Model saved to {OUTPUT_PATH}")
