from __future__ import annotations

import numpy as np

from activations import ActivationFunction, sigmoid


class Perceptron:
    """A single neuron with a configurable activation function.

    Args:
        inputs: Number of inputs, not counting the bias.
        bias: Bias value added to the weighted sum.
        activation: Activation function applied to the weighted sum.
    """

    def __init__(
        self,
        inputs: int,
        bias: float = 1.0,
        activation: ActivationFunction = sigmoid,
    ) -> None:
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias
        self._activation = activation

    def run(self, x: list[float]) -> float:
        """Compute the neuron output for input vector x."""
        x_sum = np.dot(np.append(x, self.bias), self.weights)
        return self._activation(x_sum)

    def set_weights(self, w_init: list[float]) -> None:
        """Set the neuron weights from a list."""
        self.weights = np.array(w_init)
