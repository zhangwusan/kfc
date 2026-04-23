from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from cobra.core.factory import BaseFactory


class BaseKernel(ABC):
    """
    Multi-view kernel with explicit tensor fusion.

    D shape:
        - (n, n)        -> single view
        - (k, n, n)     -> multi-view

    alpha shape:
        - (k,)          -> view weights

    Fusion:
        D_fused = sum_k alpha_k * D_k
                 = tensordot(alpha, D, axes=(0, 0))
    """

    def __init__(self, theta=None, **kwargs):
        self.theta = theta
        self.set_params(**kwargs)

    # ---------------- representation ----------------
    def __repr__(self):
        attrs = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
        return f"{self.__class__.__name__}({attrs})"

    # ---------------- parameters ----------------
    def set_params(self, theta=None, **kwargs):
        if theta is not None:
            self.theta = np.asarray(theta, dtype=float)

        for k, v in kwargs.items():
            setattr(self, k, v)

        return self

    def get_params(self):
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    # ---------------- tensor fusion (KEY PART) ----------------
    def fuse(self, D: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Convert multi-view distances -> single distance matrix
        using weighted linear fusion.
        """
        D = np.asarray(D, dtype=float)

        if D.ndim == 2:
            return D  # already single view

        theta = np.asarray(theta, dtype=float)

        if D.shape[0] != len(theta):
            raise ValueError(
                f"Mismatch: {D.shape[0]} views vs {len(theta)} weights"
            )

        # (k, n, n) ⊗ (k,) -> (n, n)
        return np.tensordot(theta, D, axes=(0, 0))

    # ---------------- main API ----------------
    def __call__(self, D, theta=None):
        D = np.asarray(D, dtype=float)

        if theta is not None:
            self.theta = np.asarray(theta, dtype=float)

        if self.theta is None:
            raise ValueError("theta must be provided")

        D_fused = self.fuse(D, self.theta)

        # kernel step (single matrix now)
        return self.compute(D_fused)

    # ---------------- core ----------------
    @abstractmethod
    def compute(self, D: np.ndarray) -> np.ndarray:
        """
        Input:
            D: (n, n) fused distance matrix

        Output:
            K: (n, n) kernel matrix
        """
        raise NotImplementedError

class KernelFactory(BaseFactory):
    """
    Factory for creating kernel instances.
    """
    registry = {}
