"""Concrete kernels for hard and soft consensus weighting."""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory


@KernelFactory.register("indicator", "hard")
class IndicatorKernel(BaseKernel):

    def compute(self, D, params):
        threshold = params.get("theta", 0.5)

        # D: (k, n, n)
        return (D < threshold).astype(float)


@KernelFactory.register("rbf", "gaussian")
class RBFKernel(BaseKernel):

    def compute(self, D, params):
        gamma = params.get("gamma", 1.0)

        # apply per-view
        return np.exp(-gamma * (D ** 2))

@KernelFactory.register("laplace")
class LaplaceKernel(BaseKernel):

    def compute(self, D, params):
        gamma = params.get("gamma", 1.0)

        return np.exp(-gamma * np.abs(D))
