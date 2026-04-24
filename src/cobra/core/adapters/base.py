from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from cobra.core.factory import BaseFactory


class BaseKernelAdapter(ABC):
    """
    Adapter that unifies one or more distance matrices
    into a kernel-ready representation.
    """

    def __init__(self, **kwargs):
        self.params = dict(kwargs)
        self.set_params(**kwargs)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        self.params.update(params)
        return self

    def get_params(self, deep=True):
        return dict(self.params)

    @abstractmethod
    def transform(self, *distances: np.ndarray) -> np.ndarray:
        """
        Pure transformation of distance matrices -> kernel input.
        """
        raise NotImplementedError


class KernelAdapterFactory(BaseFactory):
    """Factory for BaseKernelAdapter implementations."""
    pass