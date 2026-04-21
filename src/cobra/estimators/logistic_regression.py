from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from gradientcobra.estimators.base import BaseEstimator
from gradientcobra.factories.estimator import EstimatorFactory

@EstimatorFactory.register("logistic_regression", "logreg")
class LogisticRegressionEstimator(BaseEstimator):
    def __init__(self, max_iter=2000, **kwargs):
        super().__init__(max_iter=max_iter, **kwargs)
        self.model = LogisticRegression(max_iter=max_iter)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
