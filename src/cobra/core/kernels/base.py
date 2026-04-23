from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from cobra.core.factory import BaseFactory

class BaseKernel(ABC):
    """
    Two-level parameter system:

    1. alpha: fusion weights (view-level)
    2. params: kernel parameters
    """

    def __init__(self, alpha=None, **params):
        self.alpha = None
        self.params = {}
        self.set_params(alpha=alpha, **params)

    # ---------------- parameters ----------------
    def set_params(self, alpha=None, **params):
        if alpha is not None:
            self.alpha = np.asarray(alpha, dtype=float)

        for k, v in params.items():
            self.params[k] = v

        return self

    def get_params(self):
        return {
            "alpha": self.alpha,
            **self.params
        }

    # ---------------- fusion (ALPHA ONLY) ----------------
    def fuse(self, D: np.ndarray) -> np.ndarray:
        if D.ndim == 2:
            return D

        if self.alpha is None:
            raise ValueError("alpha must be provided for multi-view fusion")

        alpha = np.asarray(self.alpha, dtype=float)

        if D.shape[0] != len(alpha):
            raise ValueError("alpha shape mismatch with D")

        return np.tensordot(alpha, D, axes=(0, 0))

    # ---------------- main API ----------------
    def __call__(self, D, **params):
        if params:
            self.set_params(**params)

        D = np.asarray(D, dtype=float)
        D_fused = self.fuse(D)

        return self.compute(D_fused, self.params)

    @abstractmethod
    def compute(self, D, params):
        raise NotImplementedError

class KernelFactory(BaseFactory):
    """Registry-backed factory for kernel implementations."""
