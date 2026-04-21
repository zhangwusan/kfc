
from __future__ import annotations
import numpy as np
from cobra.factories.kernel import KernelFactory
from cobra.kernels.base import BaseKernel


@KernelFactory.register("cosine")
class CosineKernel(BaseKernel):
    """
    Cosine similarity-based kernel.
    """

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.dot(x, y) / (
            np.linalg.norm(x) * np.linalg.norm(y) + 1e-8
        )
