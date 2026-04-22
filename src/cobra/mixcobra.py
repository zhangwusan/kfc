"""MixCOBRA implementation built on modular core components."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator as SkBaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from cobra.core import (
	AggregatorFactory,
	DistanceFactory,
	KernelFactory,
	LossFactory,
	SpaceProjectorFactory,
)
from cobra.core.aggregators.base import BaseAggregator
from cobra.core.distances.base import BaseDistance
from cobra.core.estimators.base import BaseEstimator, EstimatorFactory
from cobra.core.kernels.base import BaseKernel
from cobra.core.losses.base import BaseLoss
from cobra.core.optimizers.base import BaseOptimizer, OptimizerFactory
from cobra.core.spaces.base import BaseSpaceProjector
from cobra.core.splitters.base import BaseDataSplitter, SplitterFactory

class MixCOBRARegressor(ABC, SkBaseEstimator, RegressorMixin):
	"""
	MixCOBRARegressor
	"""
	def __init__(
		self,
		estimators: list[str | BaseEstimator] | None = None,
		estimators_params: dict[str, Any] | None = None,
		splitter: str = "holdout",
		splitter_params: dict[str, Any] | None = None,
		distance: str = "euclidean",
		distance_params: dict[str, Any] | None = None,
		kernel: str = "rbf",
		kernel_params: dict[str, Any] | None = None,
		aggregator: str = "weighted_mean",
		aggregator_params: dict[str, Any] | None = None,
		loss: str = "mse",
		loss_params: dict[str, Any] | None = None,
		optimizer: str = "grad",
		optimizer_params: dict[str, Any] | None = None,

		alpha_list: np.ndarray | None = None,
		beta_list: np.ndarray | None = None,
		one_parameter: bool = False,
		random_state: int | None = None
	):
		self.estimators = estimators
		self.estimators_params = estimators_params
		self.splitter = splitter
		self.splitter_params = splitter_params
		self.distance = distance
		self.distance_params = distance_params
		self.kernel = kernel
		self.kernel_params = kernel_params
		self.aggregator = aggregator
		self.aggregator_params = aggregator_params
		self.loss = loss
		self.loss_params = loss_params
		self.optimizer = optimizer
		self.optimizer_params = optimizer_params

		self.alpha_list = alpha_list
		self.beta_list = beta_list
		self.one_parameter = one_parameter
		self.random_state = random_state
        
	
	def _resolve_fit_split_context(self, X, y, X_l, y_l):
		"""
		Returns: X_k, y_k, X_l, y_l, iloc_k, iloc_l, as_predictions
			X_k, y_k: training set for base estimators
			X_l, y_l: aggregation set for COBRA
			iloc_k, iloc_l: indices of X_k, X_l in original X
		"""
		X, y = check_X_y(X, y)
		if X_l is not None and y_l is not None:
			X_l, y_l = check_X_y(X_l, y_l)
			X_k_, X_l_ = X, X_l
			y_k_, y_l_ = y, y_l
			iloc_l, iloc_k = np.arange(len(y_l_)), np.arange(len(y))
			self.as_predictions_ = True
		else:
			split_params = dict(self.splitter_params or {})
			split_params.setdefault("random_state", self.random_state)
			splitter: BaseDataSplitter = SplitterFactory.create(
				self.splitter,
				**split_params
			)
			iloc_k, iloc_l = splitter.split(X, y)
			X_k_, y_k_ = X[iloc_k], y[iloc_k]
			X_l_, y_l_ = X[iloc_l], y[iloc_l]
			self.as_predictions_ = False

		return X_k_, y_k_, X_l_, y_l_, iloc_k, iloc_l

	
	def _fit_estimators(self, X_k: np.ndarray, y_k: np.ndarray):
		"""
		Build and fit base estimators.
		"""

		default_estimators = [
			"linear",
            "ridge",
            "lasso",
            "knn",
            "random_forest",
            "svm",
		]

		estimators = self.estimators or default_estimators

		machines = []

		for est in estimators:
			if isinstance(est, str):
				params = (self.estimators_params or {}).get(est, {})
				model = EstimatorFactory.create(est, **params)
			elif isinstance(est, BaseEstimator):
				model = est
			else:
				raise ValueError(
					f"Invalid estimator: {type(est)}. "
					f"Expected str or BaseEstimator. "
					f"Available: {EstimatorFactory.available()}"
				)

			model.fit(X_k, y_k)
			machines.append(model)

		return machines

	def _prediction_matrix(self, X: np.ndarray):
		if self.as_predictions_:
			return X

		cols = []
		for est in self.base_estimators_:
			cols.append(np.asarray(est.predict(X)).reshape(-1, 1))
		return np.hstack(cols)

	def _space_projector(self, X, pred_matrix, alpha, beta):
		params = {"alpha" : alpha, "beta" : beta}
		projector: BaseSpaceProjector = SpaceProjectorFactory.create("mixcobra", **params)
		return projector.transform(X, pred_matrix)

	def fit(
		self,
		X: np.ndarray,
		y: np.ndarray,
		X_l: np.ndarray | None = None,
		y_l: np.ndarray | None = None
	):
		
		(
			self.X_k_, self.y_k_,
			self.X_l_, self.y_l_,
			self.iloc_k_, self.iloc_l_
		) = self._resolve_fit_split_context(X, y, X_l, y_l)

		if not self.as_predictions_:
			self.base_estimators_ = self._fit_estimators(self.X_k_, self.y_k_)
			pred_l = self._prediction_matrix(self.X_l_)
			self.z_l_ = self._space_projector(self.X_l_, pred_l, 1.0, 1.0)
		else:
			self.z_l_ = self.X_l_
		
		self.distance_ : BaseDistance = DistanceFactory.create(self.distance, **(self.distance_params or {}))
		self.kernel_ : BaseKernel = KernelFactory.create(self.kernel, **(self.kernel_params or {}))
		self.aggregator_ : BaseAggregator = AggregatorFactory.create(self.aggregator, **(self.aggregator_params or {}))
		self.loss_ : BaseLoss = LossFactory.create(self.loss, **(self.loss_params or {}))
		self.optimizer_ : BaseOptimizer = OptimizerFactory.create(self.optimizer, **(self.optimizer_params or {}))
        
		return self
		
	def predict(self, X: np.ndarray) -> np.ndarray:
		check_is_fitted(self)
		X = check_array(X)

		if not self.as_predictions_:
			pred_x = self._prediction_matrix(X)
			z_x = self._space_projector(X, pred_x, 1.0, 1.0)
		else:
			z_x = X
		
		outputs = np.empty(z_x.shape[0], dtype=float)

		for i, row in enumerate(z_x):
			d = self.distance_.pairwise(row, self.z_l_)
			w = self.kernel_(d)

			w[i] = 0.0
			if np.allclose(w.sum(), 0.0):
				outputs[i] = np.mean(self.y_l_)
			else:
				outputs[i] = self.aggregator_.aggregate(self.y_l_, w)

		return outputs