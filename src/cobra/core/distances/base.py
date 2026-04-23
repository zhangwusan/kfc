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
    def __call__(self, x, y):
        pass

class DistanceFactory(BaseFactory):
    pass