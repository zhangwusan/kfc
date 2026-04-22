"""Base interface for distance-to-weight kernel mappings."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseKernel(ABC):
    """Transform distances into non-negative influence weights."""

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self) -> str:
        attrs = {key: value for key, value in self.__dict__.items() if not key.startswith("_")}
        return f"{self.__class__.__name__}({attrs})"

    @abstractmethod
    def __call__(self, distances: ArrayLike) -> np.ndarray:
        """Compute kernel weights from a 1D array of distances."""
        raise NotImplementedError


class KernelFactory(BaseFactory):
    """Registry-backed factory for kernel implementations."""
