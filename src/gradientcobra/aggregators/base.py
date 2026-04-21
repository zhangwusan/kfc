"""
Aggregation Module
Defines how weighted predictions are converted into final outputs.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAggregator(ABC):
    """
    Abstract aggregation function.

    Maps:

        weights + labels -> prediction

    This is the final decision layer in COBRA-style systems.
    """

    @abstractmethod
    def aggregate(self, weights: np.ndarray, y: np.ndarray):
        pass
