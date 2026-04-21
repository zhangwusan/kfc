"""
Aggregation Module
Defines how weighted predictions are converted into final outputs.
"""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from cobra.core.factory import BaseFactory


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

class AggregatorFactory(BaseFactory):
    """
    Factory for managing aggregation components.

    This class inherits all functionality from BaseFactory:
    - register() decorator for adding aggregators
    - create() for instantiating aggregators by name
    - available() for listing registered aggregators
    """

    _registry: Dict[str, BaseAggregator] = {}
