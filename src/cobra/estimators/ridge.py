from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from cobra.estimators.base import BaseEstimator, EstimatorFactory
@EstimatorFactory.register("ridge")
class RidgeEstimator(BaseEstimator):
    """
    L2-regularized linear regression (stable estimator).
    """

    def __init__(self, alpha=1.0):
        super().__init__(alpha=alpha)

    def fit(self, X, y: np.ndarray, **kwargs):
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X, y)
        return self

    def predict(self, X, **kwargs) -> np.ndarray:
        return self.model_.predict(X)
