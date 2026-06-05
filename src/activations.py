from __future__ import annotations

from typing import Callable

import numpy as np

ActivationFunction = Callable[[float], float]


def sigmoid(x: float) -> float:
    """Compute the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))
