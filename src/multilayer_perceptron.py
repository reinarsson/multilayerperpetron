from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from activations import ActivationFunction, sigmoid
from perceptron import Perceptron


class MultiLayerPerceptron:
    """A fully connected feedforward neural network trained via backpropagation.

    Args:
        layers: Number of neurons per layer, including input and output layers.
        bias: Bias value shared across all neurons.
        eta: Learning rate.
        activation: Activation function applied by each neuron.
    """

    def __init__(
        self,
        layers: list[int],
        bias: float = 1.0,
        eta: float = 0.5,
        activation: ActivationFunction = sigmoid,
    ) -> None:
        self.layers = np.array(layers, dtype=object)
        self.bias = bias
        self.eta = eta
        self._activation = activation

        self.network: list = []
        self.values: list = []
        self.d: list = []

        for i in range(len(self.layers)):
            self.values.append([0.0] * self.layers[i])
            self.d.append([0.0] * self.layers[i])
            self.network.append([])
            if i > 0:
                for _ in range(self.layers[i]):
                    self.network[i].append(
                        Perceptron(
                            inputs=self.layers[i - 1],
                            bias=self.bias,
                            activation=self._activation,
                        )
                    )

        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.values], dtype=object)
        self.d = np.array([np.array(x) for x in self.d], dtype=object)

    def set_weights(self, w_init: list[list[list[float]]]) -> None:
        """Set all weights from a 3D list (layer, neuron, weight)."""
        for i, layer_weights in enumerate(w_init):
            for j, neuron_weights in enumerate(layer_weights):
                self.network[i + 1][j].set_weights(neuron_weights)

    def run(self, x: list[float]) -> np.ndarray:
        """Forward pass. Returns the output layer values."""
        self.values[0] = np.array(x, dtype=object)
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i - 1])
        return self.values[-1]

    def bp(self, x: list[float], y: list[float]) -> float:
        """Run one forward pass and backpropagation step.

        Args:
            x: Input vector.
            y: Target output vector.

        Returns:
            Mean squared error for this sample.
        """
        x = np.array(x, dtype=object)
        y = np.array(y, dtype=object)

        outputs = self.run(x)

        error = y - outputs
        mse = sum(error ** 2) / self.layers[-1]

        self.d[-1] = outputs * (1 - outputs) * error

        for i in reversed(range(1, len(self.network) - 1)):
            for h in range(len(self.network[i])):
                fwd_error = sum(
                    self.network[i + 1][k].weights[h] * self.d[i + 1][k]
                    for k in range(self.layers[i + 1])
                )
                self.d[i][h] = self.values[i][h] * (1 - self.values[i][h]) * fwd_error

        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                for k in range(self.layers[i - 1] + 1):
                    if k == self.layers[i - 1]:
                        delta = self.eta * self.d[i][j] * self.bias
                    else:
                        delta = self.eta * self.d[i][j] * self.values[i - 1][k]
                    self.network[i][j].weights[k] += delta

        return mse

    def save(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save weights to a .npz file and a companion .json sidecar.

        The .json sidecar contains model parameters, the activation function
        name, and any extra metadata (e.g. training stats, accuracy) passed
        via the metadata argument.

        Args:
            path: Destination .npz file path (e.g. 'output/mnist.npz').
            metadata: Optional dict with training statistics and accuracy to
                include in the sidecar (e.g. mse_per_epoch, accuracy).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        weights = {
            f"layer_{i}_neuron_{j}": self.network[i][j].weights
            for i in range(1, len(self.network))
            for j in range(self.layers[i])
        }
        np.savez(path, layers=self.layers, **weights)

        sidecar = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "layers": self.layers.tolist(),
                "bias": self.bias,
                "eta": self.eta,
                "activation": self._activation.__name__,
                "total_params": sum(
                    len(self.network[i][j].weights)
                    for i in range(1, len(self.network))
                    for j in range(self.layers[i])
                ),
            },
        }
        if metadata:
            sidecar.update(metadata)

        sidecar_path = path.with_suffix(".json")
        sidecar_path.write_text(json.dumps(sidecar, indent=2))

    @classmethod
    def load(cls, path: str | Path, activation: ActivationFunction = sigmoid) -> MultiLayerPerceptron:
        """Load a model from a .npz file saved by save().

        Args:
            path: Path to the .npz file.
            activation: Activation function to use (must match what was used during training).

        Returns:
            A MultiLayerPerceptron with restored weights.
        """
        data = np.load(Path(path), allow_pickle=True)
        layers = data["layers"].tolist()
        mlp = cls(layers=layers, activation=activation)
        for i in range(1, len(mlp.network)):
            for j in range(mlp.layers[i]):
                mlp.network[i][j].weights = data[f"layer_{i}_neuron_{j}"]
        return mlp
