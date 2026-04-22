"""Estimator wrappers used by the consensus expert pool."""

from .base import BaseEstimator
from .base import EstimatorFactory
from .builtin import (
    MeanRegressor,
    LinearRegressorEstimator,
    RidgeRegressorEstimator,
    LassoRegressorEstimator,
    KNNRegressorEstimator,
    RandomForestRegressorEstimator,
    SVMRegressorEstimator,
)

__all__ = [
	"BaseEstimator",
	"EstimatorFactory",
	"MeanRegressor",
	"LinearRegressorEstimator",
	"RidgeRegressorEstimator",
	"LassoRegressorEstimator",
	"KNNRegressorEstimator",
	"RandomForestRegressorEstimator",
	"SVMRegressorEstimator",
]