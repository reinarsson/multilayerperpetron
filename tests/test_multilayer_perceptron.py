from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from multilayer_perceptron import MultiLayerPerceptron


@pytest.fixture
def mlp() -> MultiLayerPerceptron:
    return MultiLayerPerceptron(layers=[2, 3, 1])


def test_network_layer_count(mlp: MultiLayerPerceptron):
    assert len(mlp.network) == 3


def test_input_layer_has_no_neurons(mlp: MultiLayerPerceptron):
    assert len(mlp.network[0]) == 0


def test_hidden_layer_neuron_count(mlp: MultiLayerPerceptron):
    assert len(mlp.network[1]) == 3


def test_output_layer_neuron_count(mlp: MultiLayerPerceptron):
    assert len(mlp.network[2]) == 1


def test_run_output_shape(mlp: MultiLayerPerceptron):
    output = mlp.run([0.5, 0.5])
    assert len(output) == 1


def test_run_output_between_zero_and_one(mlp: MultiLayerPerceptron):
    output = mlp.run([0.5, 0.5])
    assert all(0.0 < v < 1.0 for v in output)


def test_bp_returns_float(mlp: MultiLayerPerceptron):
    mse = mlp.bp([0.5, 0.5], [1.0])
    assert isinstance(float(mse), float)


def test_bp_mse_is_non_negative(mlp: MultiLayerPerceptron):
    mse = mlp.bp([0.5, 0.5], [1.0])
    assert float(mse) >= 0.0


def test_bp_reduces_error_over_time():
    mlp = MultiLayerPerceptron(layers=[2, 4, 1], eta=0.5)
    first_mse = mlp.bp([1.0, 0.0], [1.0])
    for _ in range(200):
        mse = mlp.bp([1.0, 0.0], [1.0])
    assert float(mse) < float(first_mse)


def test_set_weights_updates_network():
    mlp = MultiLayerPerceptron(layers=[2, 2, 1])
    w_init = [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],   # hidden layer: 2 neurons, 2 inputs + bias
        [[7.0, 8.0, 9.0]],                       # output layer: 1 neuron, 2 inputs + bias
    ]
    mlp.set_weights(w_init)
    np.testing.assert_array_equal(mlp.network[1][0].weights, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(mlp.network[2][0].weights, [7.0, 8.0, 9.0])


def test_xor_learns():
    """MLP should be able to learn XOR with enough epochs."""
    mlp = MultiLayerPerceptron(layers=[2, 4, 1], eta=0.5)
    for _ in range(5000):
        mlp.bp([0, 0], [0])
        mlp.bp([0, 1], [1])
        mlp.bp([1, 0], [1])
        mlp.bp([1, 1], [0])
    assert mlp.run([0, 0])[0] < 0.1
    assert mlp.run([0, 1])[0] > 0.9
    assert mlp.run([1, 0])[0] > 0.9
    assert mlp.run([1, 1])[0] < 0.1
