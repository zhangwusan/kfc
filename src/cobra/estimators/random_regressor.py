
from __future__ import annotations

import numpy as np

from cobra.estimators.base import BaseEstimator, EstimatorFactory

@EstimatorFactory.register("random_regressor", "noise")
class RandomRegressor(BaseEstimator):
    """
    Returns random predictions based on training label distribution.

    Purpose:
    - stress test aggregation robustness
    - simulate weak base learners
    """

    def fit(self, X, y: np.ndarray, **kwargs):
        self.y_mean_ = np.mean(y)
        self.y_std_ = np.std(y)
        return self

    def predict(self, X, **kwargs) -> np.ndarray:
        return np.random.normal(
            loc=self.y_mean_,
            scale=self.y_std_ + 1e-8,
            size=len(X)
        )
