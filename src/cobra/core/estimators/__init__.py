"""Estimator wrappers used by the consensus expert pool."""

from .base import BaseEstimator
from .base import EstimatorFactory
from .builtin import MeanRegressor, SklearnRegressor

__all__ = [
	"BaseEstimator",
	"EstimatorFactory",
	"SklearnRegressor",
	"MeanRegressor",
]