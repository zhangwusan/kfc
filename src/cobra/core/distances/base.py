from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseDistance(ABC):
    """
    Unified distance interface.

    -------------------------
    Outputs:
    -------------------------
    - pairwise: (query, X) → (n,)
    - matrix:   (X, X) → (n, n)
    - tensor:   list[X_k] → (k, n, n)
    """

    # -------------------------
    # 1. Pairwise distance
    # -------------------------
    @abstractmethod
    def pairwise(self, query: ArrayLike, candidates: ArrayLike) -> np.ndarray:
        """
        Distance from one query to multiple candidates.

        Returns:
            (n_candidates,)
        """
        raise NotImplementedError

    # -------------------------
    # 2. Full distance matrix
    # -------------------------
    @abstractmethod
    def matrix(self, X: ArrayLike) -> np.ndarray:
        """
        Full pairwise distance matrix.

        Returns:
            (n, n)
        """
        raise NotImplementedError

    # -------------------------
    # 3. Multi-view tensor
    # -------------------------
    def tensor(self, X_list: list[ArrayLike]) -> np.ndarray:
        """
        Multi-view distance tensor.

        Returns:
            (k, n, n)
        """
        return np.stack([self.matrix(X) for X in X_list])


# -------------------------
# Factory
# -------------------------
class DistanceFactory(BaseFactory):
    """Registry-backed factory for distance implementations."""
