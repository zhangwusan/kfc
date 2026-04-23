from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from cobra.core.factory import BaseFactory


class BaseKernel(ABC):
    """
    kernel fusion base class.

    Assumption:
        D is ALWAYS a tensor of shape (k, n, n)

    Model:
        K = Σ α_k * compute(D_k, params)
    """

    def __init__(self, alpha: np.ndarray, **params):
        self.alpha = np.asarray(alpha, dtype=float)
        self.params = dict(params)

    # ---------------- parameters ----------------
    def set_params(self, alpha: np.ndarray | None = None, **params):
        if alpha is not None:
            self.alpha = np.asarray(alpha, dtype=float)

        self.params.update(params)
        return self

    def get_params(self):
        return {"alpha": self.alpha, **self.params}

    # ---------------- fusion ----------------
    def fuse(self, K: np.ndarray) -> np.ndarray:
        """
        Fuse per-view kernel matrices.

        Parameters
        ----------
        K : np.ndarray
            Kernel tensor of shape (k, n, n)

        Returns
        -------
        np.ndarray
            Fused kernel matrix (n, n)
        """
        K = np.asarray(K, dtype=float)

        if K.ndim != 3:
            raise ValueError(f"Expected (k,n,n), got {K.shape}")

        if K.shape[0] != len(self.alpha):
            raise ValueError(
                f"alpha mismatch: {len(self.alpha)} vs {K.shape[0]}"
            )

        return np.tensordot(self.alpha, K, axes=(0, 0))

    # ---------------- main API ----------------
    def __call__(self, D: np.ndarray, **params) -> np.ndarray:
        """
        Compute fused kernel matrix from distance tensor.
        """
        if params:
            self.set_params(**params)

        D = np.asarray(D, dtype=float)

        if D.ndim != 3:
            raise ValueError(f"D must be (k,n,n), got {D.shape}")

        K_views = self.compute(D, self.params)  # (k,n,n)

        return self.fuse(K_views)

    # ---------------- core ----------------
    @abstractmethod
    def compute(self, D: np.ndarray, params: dict) -> np.ndarray:
        """
        Compute per-view kernels.

        Must return:
            (k, n, n)
        """
        raise NotImplementedError


class KernelFactory(BaseFactory):
    """Registry for kernel implementations."""
    pass
