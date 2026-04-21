
from __future__ import annotations

import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from cobra.estimators.base import BaseEstimator, EstimatorFactory

@EstimatorFactory.register("knn", "nearest_neighbors")
class KNNEstimator(BaseEstimator):
    """
    k-Nearest Neighbors regression estimator.
    """

    def __init__(self, k=5):
        super().__init__(k=k)

    def fit(self, X, y: np.ndarray, **kwargs):
        self.model_ = KNeighborsRegressor(n_neighbors=self.k)
        self.model_.fit(X, y)
        return self

    def predict(self, X, **kwargs) -> np.ndarray:
        return self.model_.predict(X)
