
from __future__ import annotations
from abc import ABC
from typing import Dict

from cobra.core.factory import BaseFactory


class BaseLoss(ABC):
    def __init__(self):
        pass
    
    def __call__(self, y_true, y_pred, weight=None, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

class LossFactory(BaseFactory):
    """
    Factory for managing loss components.

    This class inherits all functionality from BaseFactory:
    - register() decorator for adding losses
    - create() for instantiating losses by name
    - available() for listing registered losses
    """

    _registry: Dict[str, BaseLoss] = {}
