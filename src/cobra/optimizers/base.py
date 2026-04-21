"""
Optimizer Module
----------------
Responsible for optimizing hyperparameters (e.g., bandwidth h)
given a black-box loss function.
"""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from cobra.core.factory import BaseFactory

try:
    from tqdm import tqdm, trange
except ImportError:
    # fallback dummy progress bars
    def tqdm(iterable=None, **kwargs):
        return iterable
    def trange(*args, **kwargs):
        return range(*args)
class BaseOptimizer(ABC):
    """
    Base class for all optimizers.

    Contract
    --------
    - Input: loss function L(h)
    - Output: optimized parameter h*
    """

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def step(self, loss_fn, init_value: float):
        """
        Perform optimization and return optimal parameter.
        """
        pass

class OptimizerFactory(BaseFactory):
    """
    Factory for managing optimizer components.

    This class inherits all functionality from BaseFactory:
    - register() decorator for adding optimizers
    - create() for instantiating optimizers by name
    - available() for listing registered optimizers
    """

    _registry: Dict[str, BaseOptimizer] = {}
