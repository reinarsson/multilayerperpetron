from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from activations import sigmoid
from perceptron import Perceptron

N_INPUTS = 3


@pytest.fixture
def perceptron() -> Perceptron:
    p = Perceptron(inputs=N_INPUTS)
    p.set_weights([0.5, -0.5, 0.3, 0.1])  # 3 inputs + 1 bias weight
    return p


def test_weights_shape_on_init():
    p = Perceptron(inputs=N_INPUTS)
    assert p.weights.shape == (N_INPUTS + 1,)


def test_weights_initialised_in_range():
    p = Perceptron(inputs=10)
    assert np.all(p.weights >= -1) and np.all(p.weights <= 1)


def test_set_weights():
    p = Perceptron(inputs=2)
    p.set_weights([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(p.weights, [1.0, 2.0, 3.0])


def test_run_returns_sigmoid_output(perceptron: Perceptron):
    x = [1.0, 0.0, 1.0]
    result = perceptron.run(x)
    expected = sigmoid(np.dot(np.append(x, perceptron.bias), perceptron.weights))
    assert result == pytest.approx(expected)


def test_run_output_between_zero_and_one(perceptron: Perceptron):
    result = perceptron.run([1.0, 1.0, 1.0])
    assert 0.0 < result < 1.0


def test_custom_activation_is_used():
    linear = lambda x: x
    p = Perceptron(inputs=1, activation=linear)
    p.set_weights([1.0, 0.0])  # weight=1, bias_weight=0
    result = p.run([2.0])
    assert result == pytest.approx(2.0)


def test_bias_included_in_computation():
    p = Perceptron(inputs=1, bias=2.0)
    p.set_weights([0.0, 1.0])  # input weight=0, bias weight=1
    result = p.run([99.0])  # input ignored, only bias matters
    assert result == pytest.approx(sigmoid(2.0))
