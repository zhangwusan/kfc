from abc import abstractmethod
from cobra.core.factory import BaseFactory
from cobra.core.optimizers.base import BaseOptimizer


class BaseSearchOptimizer(BaseOptimizer):
    """
    Base class for black-box, combinatorial, and hyperparameter search optimizers.

    This class defines the interface for optimizers that do not rely on gradients,
    but instead explore a discrete or stochastic search space. Examples include:
    grid search, random search, evolutionary strategies, and Bayesian optimization.
    """

    @abstractmethod
    def search_space(self):
        """
        Define the parameter search space.

        Returns
        -------
        Any
            A structured representation of the search space.
            This may be a dictionary, distribution specification,
            or custom space object depending on implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        """
        Generate the next candidate parameter set from the search space.

        This method is responsible for proposing new configurations
        to evaluate in the optimization loop.

        Returns
        -------
        np.ndarray | dict
            A sampled candidate parameter configuration.
            Format depends on the optimizer implementation.
        """
        raise NotImplementedError

class SearchOptimizerFactory(BaseFactory):
    """Factory for BaseSearchOptimizer implementations."""
    pass