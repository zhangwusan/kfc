"""Concrete kernels for hard and soft consensus weighting."""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory


# =========================
# 1. Indicator (Hard COBRA)
# =========================
@KernelFactory.register("indicator", "hard")
class IndicatorKernel(BaseKernel):
    """
    K(x, x') = 1(D < theta)
    """

    def compute(self, D: np.ndarray) -> np.ndarray:
        # D is already fused (n, n)
        theta = getattr(self, "theta", None)

        if theta is None:
            raise ValueError("theta must be provided for IndicatorKernel")

        return (D < theta).astype(float)


# =========================
# 2. Gaussian / RBF
# =========================
@KernelFactory.register("rbf", "gaussian")
class RBFKernel(BaseKernel):
    """
    K = exp(-theta * D^2)
    """

    def compute(self, D: np.ndarray) -> np.ndarray:
        theta = getattr(self, "theta", None)

        if theta is None:
            raise ValueError("theta must be provided for RBFKernel")

        return np.exp(-theta * (D ** 2))


# =========================
# 3. Laplace Kernel
# =========================
@KernelFactory.register("laplace")
class LaplaceKernel(BaseKernel):
    """
    K = exp(-theta * |D|)
    """

    def compute(self, D: np.ndarray) -> np.ndarray:
        theta = getattr(self, "theta", None)

        if theta is None:
            raise ValueError("theta must be provided for LaplaceKernel")

        return np.exp(-theta * np.abs(D))
