"""Concrete kernels for hard and soft consensus weighting."""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory

@KernelFactory.register("indicator", "hard")
class IndicatorKernel(BaseKernel):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
    
    def __call__(self, D):
        return (D < self.threshold).astype(float)

@KernelFactory.register("rbf", "gaussian")
class RBFKernel(BaseKernel):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
    
    def __call__(self, D):
        return np.exp(-self.gamma * D)

@KernelFactory.register("laplace")
class LaplaceKernel(BaseKernel):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
    
    def __call__(self, D):
        return np.exp(-self.gamma * np.abs(D))
