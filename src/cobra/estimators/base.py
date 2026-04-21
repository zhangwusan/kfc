"""
Base Estimator Module

This module defines the abstract interface for all estimators used in the
COBRA-style modular learning framework.

The estimator is responsible for learning a mapping:

    X -> r(X)

where r(X) is the prediction space representation used by:
- COBRA (discrete prediction vectors)
- MixCOBRA (joint input-output aggregation)
- GradientCOBRA (prediction-space kernel learning)

All estimators must implement:
- fit(): training procedure
- predict(): inference procedure

This design ensures compatibility across all aggregation strategies.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

from cobra.core.factory import BaseFactory


class BaseEstimator(ABC):
    """
    Abstract base class for all estimators.

    This class defines the standard interface for models that generate
    predictions used in ensemble aggregation methods such as COBRA,
    MixCOBRA, and GradientCOBRA.

    The estimator is treated as a function:

        f: X -> r(X)

    where r(X) may represent:
    - class labels (COBRA)
    - regression outputs (MixCOBRA, GradientCOBRA)
    - multi-dimensional prediction vectors (ensemble space)

    Attributes
    ----------
    **kwargs : dict
        Optional hyperparameters stored as instance attributes.
    """

    def __init__(self, **kwargs):
        """
        Initialize estimator with flexible hyperparameters.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary hyperparameters (e.g., depth, learning_rate, etc.)
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        """
        String representation of estimator showing configurable parameters.

        Returns
        -------
        str
            Human-readable representation of the estimator.
        """
        attrs = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
        return f"{self.__class__.__name__}({attrs})"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Train the estimator on labeled data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        y : np.ndarray of shape (n_samples,)
            Target labels or regression values.

        Returns
        -------
        self : BaseEstimator
            Fitted estimator instance.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate predictions for input samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted values. Shape depends on estimator type:
            - (n_samples,) for scalar output
            - (n_samples, M) for multi-output / ensemble space
        """
        pass


class EstimatorFactory(BaseFactory):
    """
    Factory for managing estimator components.

    This class inherits all functionality from BaseFactory:
    - register() decorator for adding estimators
    - create() for instantiating estimators by name
    - available() for listing registered estimators
    """

    _registry: Dict[str, BaseEstimator] = {}
