

from abc import ABC, abstractmethod

from cobra.core.factory import BaseFactory


class BaseKernel(ABC):
    def __init__(self, **kwargs):
        self.params = dict(kwargs)
        for key, value in self.params.items():
            setattr(self, key, value)
        
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            self.params[key] = value
    
    def get_params(self, deep=True):
        return self.params
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class KernelFactory(BaseFactory):
    """Factory for BaseKernel implementations."""
    pass
