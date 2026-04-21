from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from gradientcobra.estimators.base import BaseEstimator
from gradientcobra.factories.estimator import EstimatorFactory


@EstimatorFactory.register("linear_regression", "linear", "ols")
class LinearRegressionEstimator(BaseEstimator):
    """
    Ordinary Least Squares Regression.
    """
    def fit(self, X, y: np.ndarray, **kwargs):
        self.model_ = LinearRegression()
        self.model_.fit(X, y)
        return self

    def predict(self, X, **kwargs) -> np.ndarray:
        return self.model_.predict(X)
