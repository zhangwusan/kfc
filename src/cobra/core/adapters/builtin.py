

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

    def __init__(self, alpha: float = 1.0, beta: float = 0.0):
        super().__init__(alpha=alpha, beta=beta)

    def transform(self, *distances: np.ndarray) -> np.ndarray:
        if len(distances) != 2:
            raise ValueError("Expected exactly 2 distance matrices")

        return self.alpha * distances[0] + self.beta * distances[1]
