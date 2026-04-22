"""Concrete estimator wrappers for the expert pool."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.dummy import DummyRegressor

from .base import BaseEstimator, EstimatorFactory


@EstimatorFactory.register("sklearn_regressor", "sklearn")
class SklearnRegressor(BaseEstimator):
    """Wrap a scikit-learn compatible regressor instance."""

    def __init__(self, estimator) -> None:
        self.estimator = clone(estimator)

    def fit(self, x: ArrayLike, y: ArrayLike) -> "SklearnRegressor":
        self.estimator.fit(x, y)
        return self

    def predict(self, x: ArrayLike):
        return self.estimator.predict(x)


@EstimatorFactory.register("mean_regressor", "dummy_mean")
class MeanRegressor(BaseEstimator):
    """Simple baseline regressor predicting the training mean."""

    def __init__(self) -> None:
        self.estimator = DummyRegressor(strategy="mean")

    def fit(self, x: ArrayLike, y: ArrayLike) -> "MeanRegressor":
        self.estimator.fit(x, y)
        return self

    def predict(self, x: ArrayLike):
        preds = self.estimator.predict(x)
        return np.asarray(preds, dtype=float)
