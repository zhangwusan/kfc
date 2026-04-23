from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from cobra.core.factory import BaseFactory


class BaseKernel(ABC):
    """
    Two-level parameter system:

    1. alpha: fusion weights (view-level)
        - REQUIRED for multi-view
        - MUST be np.ndarray of shape (k,)
    2. params: kernel parameters
    """

    def __init__(self, alpha: np.ndarray, **params):
        self.alpha: np.ndarray
        self.params: dict = {}
        self.set_params(alpha=alpha, **params)

    # ---------------- parameters ----------------
    def set_params(self, alpha: np.ndarray, **params):
        self.alpha = alpha
        for k, v in params.items():
            self.params[k] = v
        return self

    def get_params(self):
        return {
            "alpha": self.alpha,
            **self.params
        }

    # ---------------- fusion ----------------
    def fuse(self, D: np.ndarray) -> np.ndarray:
        D = np.asarray(D, dtype=float)

        # single view → no fusion
        if D.ndim in (1, 2):
            return D

        if D.shape[0] != len(self.alpha):
            raise ValueError(
                f"Mismatch: {D.shape[0]} views vs {len(self.alpha)} alpha"
            )

        return np.tensordot(self.alpha, D, axes=(0, 0))

    # ---------------- main API ----------------
    def __call__(self, D, **params):
        if params:
            self.set_params(**params)

        D = np.asarray(D, dtype=float)
        D_fused = self.fuse(D)

        return self.compute(D_fused, self.params)

    # ---------------- core ----------------
    @abstractmethod
    def compute(self, D: np.ndarray, params: dict) -> np.ndarray:
        raise NotImplementedError

class KernelFactory(BaseFactory):
    """Registry-backed factory for kernel implementations."""