"""Base interface for pairwise distance computation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseDistance(ABC):
    """Compute distances between one query vector and multiple candidates."""

    @abstractmethod
    def pairwise(self, query: ArrayLike, candidates: ArrayLike) -> np.ndarray:
        """Return a 1D array of distances for each candidate row."""
        raise NotImplementedError


class DistanceFactory(BaseFactory):
    """Registry-backed factory for distance implementations."""
