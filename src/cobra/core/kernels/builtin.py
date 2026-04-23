"""Concrete kernels for hard and soft consensus weighting."""

from __future__ import annotations
import numpy as np

from .base import BaseKernel, KernelFactory

@KernelFactory.register("indicator", "hard")
class IndicatorKernel(BaseKernel):
    def __init__(self, alpha=1, beta=0, threshold=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
    
    def __call__(self, dist1, dist2=None):
        if dist2 is None:
            score = self.alpha * dist1
        else:
            score = self.alpha * dist1 + self.beta * dist2
        return (score < self.threshold).astype(float)

@KernelFactory.register("rbf", "gaussian")
class RBFKernel(BaseKernel):
    def __init__(self, alpha=1, beta=0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def __call__(self, dist1, dist2=None):
        if dist2 is None:
            score = self.alpha * dist1
        else:
            score = self.alpha * dist1 + self.beta * dist2
        return np.exp(-self.gamma * score)

@KernelFactory.register("laplace")
class LaplaceKernel(BaseKernel):
    def __init__(self, alpha=1, beta=0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def __call__(self, dist1, dist2=None):
        if dist2 is None:
            score = self.alpha * dist1
        else:
            score = self.alpha * dist1 + self.beta * dist2
        return np.exp(-self.gamma * np.abs(score))
