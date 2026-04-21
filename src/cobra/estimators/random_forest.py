from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from cobra.estimators.base import BaseEstimator, EstimatorFactory
@EstimatorFactory.register("random_forest", "rf")
class RandomForestEstimator(BaseEstimator):
    """
    Ensemble tree-based estimator with high nonlinearity.
    """

    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X, y: np.ndarray, **kwargs):
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X, **kwargs) -> np.ndarray:
        return self.model_.predict(X)
