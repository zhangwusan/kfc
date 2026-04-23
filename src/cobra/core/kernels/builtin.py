"""Concrete kernels for hard and soft consensus weighting."""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory


# =========================
# 1. Indicator (Hard COBRA)
# =========================
@KernelFactory.register("indicator", "hard")
class IndicatorKernel(BaseKernel):
    def compute(self, D, params):
        threshold = params.get("theta", 0.5)
        return (D < threshold).astype(float)


# =========================
# 2. Gaussian / RBF
# =========================
@KernelFactory.register("rbf", "gaussian")
class RBFKernel(BaseKernel):
    def compute(self, D, params):
        gamma = params.get("gamma", 1.0)
        return np.exp(-gamma * D**2)


# =========================
# 3. Laplace Kernel
# =========================
@KernelFactory.register("laplace")
class LaplaceKernel(BaseKernel):
    def compute(self, D, params):
        gamma = params.get("gamma", 1.0)
        return np.exp(-gamma * np.abs(D))
