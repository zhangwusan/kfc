"""
Kernel Module
Defines how distances are transformed into weights for aggregation.
"""

from abc import ABC, abstractmethod
import numpy as np


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
