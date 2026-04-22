"""Concrete kernels for hard and soft consensus weighting."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import BaseKernel, KernelFactory


def _as_1d(distances: ArrayLike) -> np.ndarray:
    """Normalize distances to a flat float array."""
    return np.asarray(distances, dtype=float).reshape(-1)


@KernelFactory.register("indicator", "hard")
class IndicatorKernel(BaseKernel):
    """Hard-threshold kernel used in classic COBRA variants."""

    def __init__(self, epsilon: float = 0.5) -> None:
        self.epsilon = float(epsilon)

    def __call__(self, distances: ArrayLike) -> np.ndarray:
        d = _as_1d(distances)
        return (d <= self.epsilon).astype(float)


@KernelFactory.register("rbf", "gaussian")
class RBFKernel(BaseKernel):
    """Gaussian radial basis kernel with bandwidth $h$."""

    def __init__(self, bandwidth: float = 1.0) -> None:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be strictly positive.")
        self.bandwidth = float(bandwidth)

    def __call__(self, distances: ArrayLike) -> np.ndarray:
        d = _as_1d(distances)
        return np.exp(-(d ** 2) / self.bandwidth)


@KernelFactory.register("laplace")
class LaplaceKernel(BaseKernel):
    """Laplace kernel with heavier tails than Gaussian."""

    def __init__(self, bandwidth: float = 1.0) -> None:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be strictly positive.")
        self.bandwidth = float(bandwidth)

    def __call__(self, distances: ArrayLike) -> np.ndarray:
        d = _as_1d(distances)
        return np.exp(-np.abs(d) / self.bandwidth)
