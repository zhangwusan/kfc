from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso

from gradientcobra.estimators.base import BaseEstimator
from gradientcobra.factories.estimator import EstimatorFactory

@EstimatorFactory.register("lasso")
class LassoEstimator(BaseEstimator):
    """
    L1-regularized linear regression (sparse model).
    """

    def __init__(self, alpha=1.0):
        super().__init__(alpha=alpha)

    def fit(self, X, y: np.ndarray, **kwargs):
        self.model_ = Lasso(alpha=self.alpha)
        self.model_.fit(X, y)
        return self

    def predict(self, X, **kwargs) -> np.ndarray:
        return self.model_.predict(X)
