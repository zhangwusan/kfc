"""
Base Distance Module
Defines interface for all distance functions used in COBRA-style models.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict

from cobra.core.factory import BaseFactory


class BaseDistance(ABC):
    """
    Abstract base class for all distance functions.
    A distance maps inputs into similarity space used by kernels:
        D: (x_i, x, r_i, r_x) -> ℝ+ or structured tuple

    This abstraction allows:
    - COBRA (binary match)
    - MixCOBRA (joint distances)
    - GradientCOBRA (prediction space only)
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"{self.__class__.__name__}({attrs})"

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

class DistanceFactory(BaseFactory):
    """
    Factory for managing distance components.

    This class inherits all functionality from BaseFactory:
    - register() decorator for adding distances
    - create() for instantiating distances by name
    - available() for listing registered distances
    """

    _registry: Dict[str, BaseDistance] = {}
