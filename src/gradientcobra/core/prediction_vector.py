"""
Prediction Vector Module

This module computes and caches prediction vectors:
    r(X) = [r1(X), ..., rM(X)]

Used as the core representation in:
- COBRA
- MixCOBRA
- GradientCOBRA
"""

from __future__ import annotations
import numpy as np


class PredictionVectorizer:
    """
    Converts raw input X into prediction-space representation.

    This is the central transformation:
        X (input space) → r(X) (prediction space)
    """

    def __init__(self, estimators):
        """
        Parameters
        ----------
        estimators : list
            List of fitted base estimators [r1, r2, ..., rM]
        """
        self.estimators = estimators

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Compute prediction vectors for dataset X.

        Returns
        -------
        np.ndarray of shape (n_samples, M)
        """
        preds = [
            est.predict(X).reshape(-1, 1)
            for est in self.estimators
        ]
        return np.hstack(preds)

    def transform_single(self, x: np.ndarray) -> np.ndarray:
        """
        Compute prediction vector for a single sample.
        """
        return np.array([est.predict([x])[0] for est in self.estimators])
