"""GradientCOBRA implementation built on modular core components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from cobra.core import (
	AggregatorFactory,
	DistanceFactory,
	EstimatorFactory,
	KernelFactory,
	LossFactory,
	OptimizerFactory,
	SplitterFactory,
)
from cobra.utils.preprocessing import data_split_overlap


@dataclass
class _RegressorSpec:
	"""Descriptor for default base learners."""

	name: str
	estimator: Any


class GradientCOBRA(BaseEstimator, RegressorMixin):
	"""Prediction-space COBRA with bandwidth optimization.

	The model follows the GradientCOBRA philosophy:
	- train a pool of regressors,
	- compare points in prediction space,
	- apply a soft kernel over distances,
	- aggregate targets with kernel-weighted averaging,
	- optimize bandwidth with a calibration objective.
	"""

	def __init__(
		self,
		estimators: list[Any] | None = None,
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
		optimizer: str = "grid",
		optimizer_params: dict[str, Any] | None = None,
		bandwidth_grid: np.ndarray | None = None,
		initial_bandwidth: float = 1.0,
		random_state: int | None = None,
	) -> None:
		self.estimators = estimators
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
		self.bandwidth_grid = bandwidth_grid
		self.initial_bandwidth = initial_bandwidth
		self.random_state = random_state

	def _default_estimators(self) -> list[_RegressorSpec]:
		"""Return the default expert pool used for regression consensus."""
		return [
			_RegressorSpec("linear", LinearRegression()),
			_RegressorSpec("ridge", Ridge(alpha=1.0, random_state=self.random_state)),
			_RegressorSpec("lasso", Lasso(alpha=0.005, random_state=self.random_state)),
			_RegressorSpec("knn", KNeighborsRegressor(n_neighbors=7)),
			_RegressorSpec("random_forest", RandomForestRegressor(n_estimators=250, random_state=self.random_state)),
			_RegressorSpec("svm", SVR(C=5.0, epsilon=0.05, kernel="rbf")),
		]

	def _resolve_estimators(self) -> list[Any]:
		"""Resolve estimator list from aliases, core wrappers, or sklearn instances."""
		defaults = {spec.name: spec.estimator for spec in self._default_estimators()}

		if self.estimators is None:
			return [clone(est) for est in defaults.values()]

		resolved: list[Any] = []
		for item in self.estimators:
			if isinstance(item, str):
				key = item.lower()
				if EstimatorFactory.contains(key):
					resolved.append(EstimatorFactory.create(key))
				elif key in defaults:
					resolved.append(clone(defaults[key]))
				else:
					raise KeyError(
						f"Unknown estimator alias '{item}'. "
						f"Available aliases: {sorted(set(defaults) | set(EstimatorFactory.available()))}."
					)
			else:
				resolved.append(clone(item))
		return resolved

	def _predict_matrix(self, x: np.ndarray, estimators: list[Any]) -> np.ndarray:
		"""Stack each estimator prediction as a column."""
		cols = [np.asarray(est.predict(x), dtype=float).reshape(-1, 1) for est in estimators]
		return np.hstack(cols)

	def _build_kernel(self, bandwidth: float):
		"""Instantiate kernel while mapping bandwidth for hard or soft kernels."""
		params = dict(self.kernel_params or {})
		if self.kernel in {"indicator", "hard"}:
			params["epsilon"] = float(bandwidth)
		else:
			params["bandwidth"] = float(bandwidth)
		return KernelFactory.create(self.kernel, **params)

	def _leave_one_out_error(self, bandwidth: float) -> float:
		"""Compute calibration leave-one-out error for a candidate bandwidth."""
		kernel = self._build_kernel(bandwidth)
		n = self.pred_agg_.shape[0]
		preds = np.empty(n, dtype=float)

		for i in range(n):
			d = self.distance_.pairwise(self.pred_agg_[i], self.pred_agg_)
			w = np.asarray(kernel(d), dtype=float)
			w[i] = 0.0

			if np.allclose(w.sum(), 0.0):
				preds[i] = float(np.mean(self.y_agg_))
			else:
				preds[i] = float(self.aggregator_.aggregate(self.y_agg_, w))

		return float(self.loss_(self.y_agg_, preds))

	def _optimize_bandwidth(self) -> float:
		"""Optimize kernel bandwidth with grid or gradient optimizer."""
		if self.optimizer == "grid" and self.bandwidth_grid is not None:
			grid = np.asarray(self.bandwidth_grid, dtype=float).reshape(-1)
			scores = np.array([self._leave_one_out_error(v) for v in grid], dtype=float)
			best_idx = int(np.argmin(scores))
			self.optimization_outputs_ = {
				"opt_method": "grid",
				"opt_bandwidth": float(grid[best_idx]),
				"opt_risk": float(scores[best_idx]),
			}
			return float(grid[best_idx])

		opt_params = dict(self.optimizer_params or {})
		optimizer = OptimizerFactory.create(self.optimizer, **opt_params)
		best = float(optimizer.optimize(self._leave_one_out_error, self.initial_bandwidth))
		self.optimization_outputs_ = {
			"opt_method": self.optimizer,
			"opt_bandwidth": best,
			"opt_risk": float(self._leave_one_out_error(best)),
		}
		return best

	def fit(
		self,
		X: np.ndarray,
		y: np.ndarray,
		X_l: np.ndarray | None = None,
		y_l: np.ndarray | None = None,
		split: float = 0.5,
		overlap: float = 0.0,
		as_predictions: bool = False,
	) -> "GradientCOBRA":
		"""Fit model with optional explicit aggregation set or prediction-space input.

		When `as_predictions=True`, `X` is treated as prediction features directly
		(and no base estimators are trained).
		"""
		X, y = check_X_y(X, y, y_numeric=True)
		y = np.asarray(y, dtype=float)
		self.as_predictions_ = bool(as_predictions)

		if X_l is not None or y_l is not None:
			if X_l is None or y_l is None:
				raise ValueError("Both X_l and y_l must be provided together.")
			X_l, y_l = check_X_y(X_l, y_l, y_numeric=True)
			self.x_train_, self.y_train_ = X, y
			self.x_agg_, self.y_agg_ = X_l, np.asarray(y_l, dtype=float)
			self.as_predictions_ = False
		elif self.as_predictions_:
			self.x_train_, self.y_train_ = None, None
			self.x_agg_, self.y_agg_ = None, y
			self.pred_agg_ = np.asarray(X, dtype=float)
		else:
			self.splitter_params = dict(self.splitter_params or {})
			if self.splitter == "overlap":
				self.splitter_params = {
					**(self.splitter_params or {}),
					"split": float(split),
					"overlap": float(overlap),
					"random_state": self.random_state,
				}
			splitter = SplitterFactory.create(self.splitter, **self.splitter_params)
			idx_train, idx_agg = splitter.split(X, y)
			self.x_train_, self.y_train_ = X[idx_train], y[idx_train]
			self.x_agg_, self.y_agg_ = X[idx_agg], y[idx_agg]

		if not self.as_predictions_:
			self.base_estimators_ = self._resolve_estimators()
			for est in self.base_estimators_:
				est.fit(self.x_train_, self.y_train_)
			self.pred_agg_ = self._predict_matrix(self.x_agg_, self.base_estimators_)

		self.distance_ = DistanceFactory.create(self.distance, **dict(self.distance_params or {}))
		self.aggregator_ = AggregatorFactory.create(self.aggregator, **dict(self.aggregator_params or {}))
		self.loss_ = LossFactory.create(self.loss, **dict(self.loss_params or {}))

		self.opt_bandwidth_ = self._optimize_bandwidth()
		self.kernel_ = self._build_kernel(self.opt_bandwidth_)
		return self

	def predict(self, X: np.ndarray, bandwidth: float | None = None) -> np.ndarray:
		"""Predict by weighted aggregation in prediction space.

		When model was fitted with `as_predictions=True`, `X` is interpreted as
		prediction-space features directly.
		"""
		check_is_fitted(self, ["pred_agg_", "distance_", "aggregator_", "opt_bandwidth_"])
		X = check_array(X)
		if self.as_predictions_:
			pred_x = np.asarray(X, dtype=float)
		else:
			check_is_fitted(self, ["base_estimators_"])
			pred_x = self._predict_matrix(X, self.base_estimators_)

		kernel = self._build_kernel(self.opt_bandwidth_ if bandwidth is None else float(bandwidth))

		outputs = np.empty(pred_x.shape[0], dtype=float)
		for i, row in enumerate(pred_x):
			d = self.distance_.pairwise(row, self.pred_agg_)
			w = np.asarray(kernel(d), dtype=float)
			outputs[i] = float(self.aggregator_.aggregate(self.y_agg_, w))
		return outputs
