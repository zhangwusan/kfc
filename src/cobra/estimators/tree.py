
from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from cobra.estimators.base import BaseEstimator, EstimatorFactory

@EstimatorFactory.register("tree", "decision_tree")
class TreeEstimator(BaseEstimator):
    """
    Decision Tree estimator for non-linear base learners.
    """
    def __init__(self, max_depth=None):
        super().__init__(max_depth=max_depth)

    def fit(self, X, y: np.ndarray, **kwargs):
        self.model_ = DecisionTreeRegressor(max_depth=self.max_depth)
        self.model_.fit(X, y)
        return self

    def predict(self, X, **kwargs) -> np.ndarray:
        return self.model_.predict(X)
