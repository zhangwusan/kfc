"""Base interface for distance-to-weight kernel mappings."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseKernel(ABC):
    """Transform distances into non-negative influence weights."""

    @abstractmethod
    def __call__(self, distances: ArrayLike) -> np.ndarray:
        """Compute kernel weights from a 1D array of distances."""
        raise NotImplementedError


class KernelFactory(BaseFactory):
    """Registry-backed factory for kernel implementations."""
