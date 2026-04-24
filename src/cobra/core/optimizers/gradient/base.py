from abc import abstractmethod
from typing import Callable

import numpy as np

from cobra.core.factory import BaseFactory
from cobra.core.optimizers.base import BaseOptimizer


class BaseGradientOptimizer(BaseOptimizer):
    """
    Base class for continuous optimization algorithms.

    This class defines the interface for gradient-based optimizers such as:
    Gradient Descent, Momentum, Adam, RMSProp, and related methods.

    Optimizers derived from this class operate on continuous parameter spaces
    and update parameters iteratively using gradient information.
    """

    @abstractmethod
    def step(
        self,
        objective: Callable[[np.ndarray], float],
        params: np.ndarray
    ) -> np.ndarray:
        """
        Perform a single optimization step.

        This method computes the gradient (or an approximation of it)
        and updates the parameters accordingly.

        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            The objective function to minimize. It maps a parameter vector
            to a scalar loss value.
        params : np.ndarray
            Current parameter vector before the update step.

        Returns
        -------
        np.ndarray
            Updated parameter vector after applying one optimization step.
        """
        raise NotImplementedError

class GradientOptimizerFactory(BaseFactory):
    """Factory for BaseGradientOptimizer implementations."""
    pass