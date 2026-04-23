from abc import ABC, abstractmethod
import numpy as np

from cobra.core.factory import BaseFactory

class BaseDistance(ABC):
    def __init__(self, **kwargs):
        self.params = dict(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
            self.params[k] = v

    def get_params(self, deep=True):
        return self.params

    @abstractmethod
    def matrix(self, x: np.ndarray, y : np.ndarray) -> np.ndarray:
        """
        Compute the distance matrix between two arrays.
        
        Parameters
        ----------
        x : array-like, shape (n_samples_x, n_features)
            First input array.
        y : array-like, shape (n_samples_y, n_features)
            Second input array.
        Returns
        -------
        D : array, shape (n_samples_x, n_samples_y)
            Distance matrix between x and y.
        """
        ...

class DistanceFactory(BaseFactory):
    pass