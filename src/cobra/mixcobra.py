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
		splitter: str = "kfold",
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
		norm_constant_x = None,
		norm_constant_y = None,
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
		self.norm_constant_x = norm_constant_x
		self.norm_constant_y = norm_constant_y
		self.one_parameter = one_parameter
		self.random_state = random_state

	
	def _resolve_norm_constants(self, X, y):
		if self.norm_constant_x is None:
			self.norm_constant_x_ = 5 / (np.max(np.abs(X), axis=0) * X.shape[1])
		else:
			self.norm_constant_x_ = self.norm_constant_x / (np.max(np.abs(X), axis=0) * X.shape[1])
		
		if self.estimators is None:
			M = 6
		else:
			M = len(self.estimators)

		if self.norm_constant_y is None:
			self.norm_constant_y_ = 5 / (np.max(np.abs(y)) * M)
		else:
			self.norm_constant_y_ = self.norm_constant_y / (np.max(np.abs(y)) * M)
        
	
	def _resolve_fit_split_context(self, X, y, X_l, y_l, pred_features=None):
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
		elif pred_features is not None:
			X_l_, y_l_ = X, y
			iloc_l, iloc_k = np.arange(len(y_l_)), np.arange(len(y))
			self.as_predictions_ = True
			self.pred_l_ = check_array(pred_features) * self.norm_constant_y_
			if self.pred_l_.shape[0] != self.y_l_.shape[0]:
				raise ValueError("Incompatible shapes between y_l and pred_features")
		else:
			splitter: BaseDataSplitter = SplitterFactory.create(
				'split_overlap',
				split_ratio=0.5,
				overlap=0.0,
				random_state=self.random_state
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
			preds = est.predict(X) * self.norm_constant_y_
			cols.append(preds.reshape(-1, 1))
		return np.hstack(cols)

	def _space_projector(self, X, pred_matrix):
		projector: BaseSpaceProjector = SpaceProjectorFactory.create("mixcobra")
		return projector.transform(X, pred_matrix)

	def _optimize_hyperparameters(self):

		folds = self.splitter_.split(self.X_l_, self.y_l_)

		if self.one_parameter:
			# Only optimize alpha, set beta to 0
			def objective(params):
				self.kernel_.set_params(alpha=params[0], beta=0.0)
				mix = np.column_stack((self.input_, self.output_))
				distance_matrix = self.distance_.matrix(mix, mix)
				K = self.kernel_(distance_matrix)
				preds = np.empty(self.y_l_.shape[0], dtype=float)

				for train_idx, val_idx in folds:
					w = K[val_idx][:, train_idx]
					y_train = self.y_l_[train_idx]
					for i in range(len(val_idx)):
						if np.allclose(w[i].sum(), 0.0):
							preds[val_idx[i]] = np.mean(y_train)
						else:
							preds[val_idx[i]] = self.aggregator_.aggregate(y_train, w[i])
				
				return self.loss_(self.y_l_, preds)

			best, histories = self.optimizer_.optimize(objective=objective, initial_value=[0.5])

			self.optimization_outputs_ = {
				"alpha": best[0],
				"beta": 0.0,
				"risk": objective(best),
				"histories" : histories
			}
		else:
			def objective(params):
				alpha, beta = params
				self.kernel_.set_params(alpha=alpha, beta=beta)
				dist_input = self.distance_.matrix(self.input_, self.input_)
				dist_output = self.distance_.matrix(self.output_, self.output_)
				K = self.kernel_(dist_input, dist_output)
				preds = np.empty(self.y_l_.shape[0], dtype=float)

				for train_idx, val_idx in folds:
					w = K[val_idx][:, train_idx]
					y_train = self.y_l_[train_idx]
					for i in range(len(val_idx)):
						if np.allclose(w[i].sum(), 0.0):
							preds[val_idx[i]] = np.mean(y_train)
						else:
							preds[val_idx[i]] = self.aggregator_.aggregate(y_train, w[i])
				return self.loss_(self.y_l_, preds)
			best, histories = self.optimizer_.optimize(objective=objective, initial_value=[0.5, 0.5])
			self.optimization_outputs_ = {
				"alpha": best[0],
				"beta": best[1],
				"risk": objective(best),
				"histories" : histories
			}


	def fit(
		self,
		X: np.ndarray,
		y: np.ndarray,
		X_l: np.ndarray | None = None,
		y_l: np.ndarray | None = None,
		pred_features: np.ndarray | None = None
	):
		
		(
			self.X_k_, self.y_k_,
			self.X_l_, self.y_l_,
			self.iloc_k_, self.iloc_l_
		) = self._resolve_fit_split_context(X, y, X_l, y_l, pred_features)

		self._resolve_norm_constants(X, y)

		if not self.as_predictions_:
			self.base_estimators_ = self._fit_estimators(self.X_k_, self.y_k_)
			self.pred_l_ = self._prediction_matrix(self.X_l_)
		
		self.distance_ : BaseDistance = DistanceFactory.create(self.distance, **(self.distance_params or {}))
		self.kernel_ : BaseKernel = KernelFactory.create(self.kernel, **(self.kernel_params or {'alpha' : 1, 'beta' : 1}))
		self.aggregator_ : BaseAggregator = AggregatorFactory.create(self.aggregator, **(self.aggregator_params or {}))
		self.loss_ : BaseLoss = LossFactory.create(self.loss, **(self.loss_params or {}))
		self.optimizer_ : BaseOptimizer = OptimizerFactory.create(self.optimizer, **(self.optimizer_params or {}))
		self.splitter_ : BaseDataSplitter = SplitterFactory.create(self.splitter, **(self.splitter_params or {"random_state": self.random_state}))
	
		self.input_, self.output_ = self._space_projector(self.X_l_, self.pred_l_)

		self._optimize_hyperparameters()
		return self
		
	def predict(
		self,
		X: np.ndarray,
		pred_X: np.ndarray,
		alpha: float | None = None,
		beta: float | None = None,
		bandwidth: float | None = None
	) -> np.ndarray:
		check_is_fitted(self)
		X = check_array(X)

		if not self.as_predictions_:
			preds = self._prediction_matrix(X)
		else:
			preds = X * self.norm_constant_x_
		
		if self.one_parameter:
			alpha = self.optimization_outputs_["alpha"]
			beta = 0.0
		else:
			alpha = self.optimization_outputs_["alpha"]
			beta = self.optimization_outputs_["beta"]
		
		# update kernel with optimize parameters
		self.kernel_.set_params(alpha=alpha, beta=beta)
		
		input, output = self._space_projector(X, preds)

		dist_input = self.distance_.matrix(input, self.input_)
		dist_output = self.distance_.matrix(output, self.output_)
		W = self.kernel_(dist_input, dist_output)

		outputs = np.empty(W.shape[0], dtype=float)

		for i in range(W.shape[0]):
			w = W[i]
			if np.allclose(w.sum(), 0.0):
				outputs[i] = np.mean(self.y_l_)
			else:
				outputs[i] = self.aggregator_.aggregate(self.y_l_, w)

		return outputs
