"""
Kernel Module
Defines how distances are transformed into weights for aggregation.
"""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from cobra.core.factory import BaseFactory


class BaseKernel(ABC):
    """
    Abstract kernel function.

    Maps distance → similarity weight:

        K: ℝ⁺ → ℝ⁺

    Used in:
    - COBRA (indicator)
    - MixCOBRA (joint kernel)
    - GradientCOBRA (smooth kernel)
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class KernelFactory(BaseFactory):
    """
    Factory for managing kernel components.

    This class inherits all functionality from BaseFactory:
    - register() decorator for adding kernels
    - create() for instantiating kernels by name
    - available() for listing registered kernels
    """

    _registry: Dict[str, BaseKernel] = {}
