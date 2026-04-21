from __future__ import annotations
from sklearn.tree import DecisionTreeClassifier

from gradientcobra.estimators.base import BaseEstimator
from gradientcobra.factories.estimator import EstimatorFactory

@EstimatorFactory.register("decision_tree", "tree")
class DecisionTreeEstimator(BaseEstimator):
    def __init__(self, max_depth=5, **kwargs):
        super().__init__(max_depth=max_depth, **kwargs)
        self.model = DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
