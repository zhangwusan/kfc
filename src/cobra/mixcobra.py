"""MixCOBRA implementation built on modular core components."""

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
	SpaceProjectorFactory,
	SplitterFactory,
)
from cobra.utils.preprocessing import data_split_overlap


@dataclass
class _RegressorSpec:
	"""Descriptor for default base learners."""

	name: str
	estimator: Any


class MixCOBRA(BaseEstimator, RegressorMixin):
	"""MixCOBRA with joint input-output space and two smoothing parameters.

	The method mixes geometric closeness in input space and agreement in
	prediction space by weighting distances with `(alpha, beta)`.
	"""

	def __init__(
		self,
		estimators: list[Any] | None = None,
		splitter: str = "holdout",
		splitter_params: dict[str, Any] | None = None,
		projector: str = "tradeoff",
		projector_params: dict[str, Any] | None = None,
		distance: str = "euclidean",
		distance_params: dict[str, Any] | None = None,
		kernel: str = "rbf",
		kernel_params: dict[str, Any] | None = None,
		aggregator: str = "weighted_mean",
		aggregator_params: dict[str, Any] | None = None,
		loss: str = "mse",
		loss_params: dict[str, Any] | None = None,
		alpha_grid: np.ndarray | None = None,
		beta_grid: np.ndarray | None = None,
		one_parameter: bool = False,
		random_state: int | None = None,
	) -> None:
		self.estimators = estimators
		self.splitter = splitter
		self.splitter_params = splitter_params
		self.projector = projector
		self.projector_params = projector_params
		self.distance = distance
		self.distance_params = distance_params
		self.kernel = kernel
		self.kernel_params = kernel_params
		self.aggregator = aggregator
		self.aggregator_params = aggregator_params
		self.loss = loss
		self.loss_params = loss_params
		self.alpha_grid = alpha_grid
		self.beta_grid = beta_grid
		self.one_parameter = one_parameter
		self.random_state = random_state

	def _default_estimators(self) -> list[_RegressorSpec]:
		"""Return default machine pool used in MixCOBRA papers."""
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

	def _project(self, x: np.ndarray, pred: np.ndarray, alpha: float, beta: float) -> np.ndarray:
		"""Project into the joint input-output consensus space."""
		params = dict(self.projector_params or {})
		params.update({"alpha": float(alpha), "beta": float(beta)})
		projector = SpaceProjectorFactory.create(self.projector, **params)
		return projector.transform(x, pred)

	def _leave_one_out_error(self, alpha: float, beta: float, bandwidth: float = 1.0) -> float:
		"""Compute leave-one-out error for a candidate `(alpha, beta)` pair."""
		z = self._project(self.x_agg_, self.pred_agg_, alpha, beta)
		n = z.shape[0]
		preds = np.empty(n, dtype=float)

		for i in range(n):
			d = self.distance_.pairwise(z[i], z)
			effective_distance = d / max(float(bandwidth), 1e-12)
			w = np.asarray(self.kernel_(effective_distance), dtype=float)
			w[i] = 0.0

			if np.allclose(w.sum(), 0.0):
				preds[i] = float(np.mean(self.y_agg_))
			else:
				preds[i] = float(self.aggregator_.aggregate(self.y_agg_, w))

		return float(self.loss_(self.y_agg_, preds))

	def _optimize_alpha_beta(self, bandwidth: float = 1.0) -> tuple[float, float]:
		"""Grid-search optimization for `(alpha, beta)` on calibration data."""
		alpha_grid = np.asarray(
			self.alpha_grid if self.alpha_grid is not None else np.linspace(1e-5, 5.0, 25),
			dtype=float,
		)
		beta_grid = np.asarray(
			self.beta_grid if self.beta_grid is not None else np.linspace(1e-5, 5.0, 25),
			dtype=float,
		)

		best_score = np.inf
		best_pair = (float(alpha_grid[0]), float(beta_grid[0]))

		for alpha in alpha_grid:
			if self.one_parameter:
				beta_candidates = np.array([alpha], dtype=float)
			else:
				beta_candidates = beta_grid

			for beta in beta_candidates:
				score = self._leave_one_out_error(float(alpha), float(beta), bandwidth=bandwidth)
				if score < best_score:
					best_score = score
					best_pair = (float(alpha), float(beta))

		self.optimization_outputs_ = {
			"opt_method": "grid",
			"opt_alpha": best_pair[0],
			"opt_beta": best_pair[1],
			"opt_risk": float(best_score),
		}
		return best_pair

	def fit(
		self,
		X: np.ndarray,
		y: np.ndarray,
		X_l: np.ndarray | None = None,
		y_l: np.ndarray | None = None,
		pred_features: np.ndarray | None = None,
		split: float = 0.5,
		overlap: float = 0.0,
		one_parameter: bool = False,
	) -> "MixCOBRA":
		"""Fit model and optimize input-output trade-off parameters.

		- If `X_l`/`y_l` are provided, `X`/`y` are used for base estimators and
		  `X_l`/`y_l` for aggregation.
		- If `pred_features` is provided, it is used directly as aggregation
		  prediction features (pretrained setting).
		"""
		X, y = check_X_y(X, y, y_numeric=True)
		y = np.asarray(y, dtype=float)
		self.one_parameter = bool(one_parameter)

		if X_l is not None or y_l is not None:
			if X_l is None or y_l is None:
				raise ValueError("Both X_l and y_l must be provided together.")
			X_l, y_l = check_X_y(X_l, y_l, y_numeric=True)
			self.x_train_, self.y_train_ = X, y
			self.x_agg_, self.y_agg_ = X_l, np.asarray(y_l, dtype=float)
			self.as_predictions_ = False
		elif pred_features is not None:
			self.x_train_, self.y_train_ = None, None
			self.x_agg_, self.y_agg_ = X, y
			self.pred_agg_ = check_array(pred_features)
			if self.pred_agg_.shape[0] != self.y_agg_.shape[0]:
				raise ValueError("pred_features rows must match the number of aggregation targets.")
			self.as_predictions_ = True
		else:
			if overlap > 0.0:
				x_k, y_k, x_l, y_l, _, _ = data_split_overlap(
					X,
					y,
					split=split,
					overlap=overlap,
					shuffle=True,
					random_state=self.random_state,
				)
				self.x_train_, self.y_train_ = x_k, y_k
				self.x_agg_, self.y_agg_ = x_l, y_l
			else:
				split_params = dict(self.splitter_params or {})
				split_params.setdefault("random_state", self.random_state)
				splitter = SplitterFactory.create(self.splitter, **split_params)
				idx_train, idx_agg = splitter.split(X, y)
				self.x_train_, self.y_train_ = X[idx_train], y[idx_train]
				self.x_agg_, self.y_agg_ = X[idx_agg], y[idx_agg]
			self.as_predictions_ = False

		if not self.as_predictions_:
			self.base_estimators_ = self._resolve_estimators()
			for est in self.base_estimators_:
				est.fit(self.x_train_, self.y_train_)
			self.pred_agg_ = self._predict_matrix(self.x_agg_, self.base_estimators_)

		self.distance_ = DistanceFactory.create(self.distance, **dict(self.distance_params or {}))
		self.aggregator_ = AggregatorFactory.create(self.aggregator, **dict(self.aggregator_params or {}))
		self.kernel_ = KernelFactory.create(self.kernel, **dict(self.kernel_params or {}))
		self.loss_ = LossFactory.create(self.loss, **dict(self.loss_params or {}))

		self.opt_alpha_, self.opt_beta_ = self._optimize_alpha_beta(bandwidth=1.0)
		self.opt_bandwidth_ = 1.0
		self.z_agg_ = self._project(self.x_agg_, self.pred_agg_, self.opt_alpha_, self.opt_beta_)
		return self

	def predict(
		self,
		X: np.ndarray,
		pred_X: np.ndarray | None = None,
		alpha: float | None = None,
		beta: float | None = None,
		bandwidth: float | None = None,
	) -> np.ndarray:
		"""Predict with weighted aggregation in joint space.

		Optional overrides:
		- `pred_X`: precomputed prediction features for `X`
		- `alpha`, `beta`: trade-off parameters
		- `bandwidth`: distance scaling before kernel weighting
		"""
		check_is_fitted(self, ["z_agg_", "distance_", "kernel_", "aggregator_", "opt_alpha_", "opt_beta_"])
		X = check_array(X)
		alpha_eff = self.opt_alpha_ if alpha is None else float(alpha)
		beta_eff = self.opt_beta_ if beta is None else float(beta)
		bandwidth_eff = self.opt_bandwidth_ if bandwidth is None else float(bandwidth)

		if pred_X is not None:
			pred_x = check_array(pred_X)
			if pred_x.shape[0] != X.shape[0]:
				raise ValueError("pred_X rows must match X rows.")
		elif getattr(self, "as_predictions_", False):
			pred_x = X
		else:
			check_is_fitted(self, ["base_estimators_"])
			pred_x = self._predict_matrix(X, self.base_estimators_)

		z_x = self._project(X, pred_x, alpha_eff, beta_eff)

		outputs = np.empty(z_x.shape[0], dtype=float)
		for i, row in enumerate(z_x):
			d = self.distance_.pairwise(row, self.z_agg_)
			effective_distance = d / max(float(bandwidth_eff), 1e-12)
			w = np.asarray(self.kernel_(effective_distance), dtype=float)
			outputs[i] = float(self.aggregator_.aggregate(self.y_agg_, w))
		return outputs
