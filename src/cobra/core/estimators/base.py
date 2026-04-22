"""Base interface for estimators participating in the expert pool."""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseEstimator(ABC):
    """Minimal estimator protocol for fit/predict compatibility."""

    @abstractmethod
    def fit(self, x: ArrayLike, y: ArrayLike) -> "BaseEstimator":
        """Fit estimator parameters on training data."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: ArrayLike):
        """Generate predictions for input samples."""
        raise NotImplementedError


class EstimatorFactory(BaseFactory):
    """Registry-backed factory for estimator implementations."""
