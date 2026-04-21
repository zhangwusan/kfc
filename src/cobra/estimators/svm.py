
from __future__ import annotations

import numpy as np
from sklearn.svm import SVR

from cobra.estimators.base import BaseEstimator, EstimatorFactory
@EstimatorFactory.register("svm", "svr", "support_vector")
class SVMEstimator(BaseEstimator):
    """
    Support Vector Regression with RBF kernel.
    """

    def __init__(self, C=1.0, epsilon=0.1, kernel="rbf"):
        super().__init__(C=C, epsilon=epsilon, kernel=kernel)

    def fit(self, X, y: np.ndarray, **kwargs):
        self.model_ = SVR(
            C=self.C,
            epsilon=self.epsilon,
            kernel=self.kernel
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X, **kwargs) -> np.ndarray:
        return self.model_.predict(X)
