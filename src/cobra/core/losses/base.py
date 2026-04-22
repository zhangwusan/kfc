"""Base interface for loss functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseLoss(ABC):
    """Compute scalar error between true and predicted targets."""

    @abstractmethod
    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """Return a scalar loss value."""
        raise NotImplementedError


class LossFactory(BaseFactory):
    """Registry-backed factory for loss implementations."""
