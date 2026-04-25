

from __future__ import annotations

import numpy as np
from cobra.core.adapters.base import BaseKernelAdapter, KernelAdapterFactory


@KernelAdapterFactory.register("gradientcobra")
class GradientCOBRAKernelAdapter(BaseKernelAdapter):

    def __init__(self, bandwidth: float = 1.0):
        super().__init__(bandwidth=bandwidth)

    def transform(self, *distances: np.ndarray) -> np.ndarray:
        if len(distances) != 1:
            raise ValueError("Expected exactly 1 distance matrix")

        return self.bandwidth * distances[0]


@KernelAdapterFactory.register("mixcobra")
class MixCOBRAKernelAdapter(BaseKernelAdapter):
    """
    MixCOBRA kernel adapter

    Logic:
        adapted_distance = alpha * d_x + beta * d_y
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        super().__init__(alpha=alpha, beta=beta)

    def transform(self, *distances: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        *distances : np.ndarray
            Expected:
                distances[0] = x_distance
                distances[1] = y_distance (optional)

        Returns
        -------
        np.ndarray
            Adapted distance matrix
        """

        if len(distances) == 0:
            raise ValueError("At least one distance matrix is required")

        if len(distances) > 2:
            raise ValueError(
                "MixCOBRA expects at most 2 distance matrices: "
                "(x_distance, y_distance)"
            )

        x_distance = distances[0]

        if len(distances) == 1:
            return self.alpha * x_distance

        y_distance = distances[1]

        if x_distance.shape != y_distance.shape:
            raise ValueError(
                "x_distance and y_distance must have the same shape"
            )

        return (
            self.alpha * x_distance
            + self.beta * y_distance
        )