"""Base optimizer interface for hyperparameter search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from cobra.core.factory import BaseFactory


class BaseOptimizer(ABC):
    """Optimize a scalar objective over a scalar parameter."""

    @abstractmethod
    def optimize(self, objective: Callable[[float], float], initial_value: float) -> float:
        """Return the best parameter value according to the objective."""
        raise NotImplementedError


class OptimizerFactory(BaseFactory):
    """Registry-backed factory for optimizer implementations."""
