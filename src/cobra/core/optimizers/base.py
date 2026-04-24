from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np

try:
    from tqdm import tqdm, trange
except ImportError:
    tqdm = lambda x, **kwargs: x
    trange = lambda x, **kwargs: range(x)

class BaseOptimizer(ABC):
    """
    Root interface for all optimizers.

    This class defines the minimal contract that all optimization
    algorithms must follow, including gradient-based, search-based,
    and hybrid optimizers.

    All optimizers must implement a callable interface that executes
    the optimization procedure and returns the final parameters
    along with optimization history or metadata.
    """

    @abstractmethod
    def __call__(
        self,
        objective: Callable[[np.ndarray], float],
        *args,
        **kwargs
    ):
        """
        Execute the optimization process on a given objective function.

        Parameters
        ----------
        objective : Callable[[np.ndarray], float]
            The function to minimize. It takes a parameter vector
            and returns a scalar loss value.
        *args :
            Optional positional arguments specific to the optimizer.
        **kwargs :
            Optional keyword arguments specific to the optimizer.

        Returns
        -------
        params : np.ndarray
            The best parameters found during optimization.
        history : Any
            Optimization trace (e.g., parameter trajectory, scores,
            or logs depending on implementation).
        """
        raise NotImplementedError