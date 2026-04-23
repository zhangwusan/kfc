"""Base optimizer interface for hyperparameter search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from cobra.core.factory import BaseFactory

try:
    from tqdm import tqdm, trange
except ImportError:
    tqdm = lambda x, **kwargs: x
    trange = lambda n, **kwargs: range(n)


class BaseOptimizer(ABC):
    """Optimize a scalar objective over a scalar parameter."""

    @abstractmethod
    def optimize(self,
        objective: Callable[[np.ndarray], float],
        initial_value: np.ndarray):
        """Return the best parameter value according to the objective."""
        raise NotImplementedError


class OptimizerFactory(BaseFactory):
    """Registry-backed factory for optimizer implementations."""
