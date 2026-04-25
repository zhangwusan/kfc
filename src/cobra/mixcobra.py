"""MixCOBRA implementation built on modular core components."""

from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator as SkBaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from cobra.core.adapters.base import BaseKernelAdapter, KernelAdapterFactory
from cobra.core.aggregators.base import AggregatorFactory, BaseAggregator
from cobra.core.distances.base import BaseDistance, DistanceFactory
from cobra.core.estimators.base import BaseEstimator, EstimatorFactory
from cobra.core.kernels.base import BaseKernel, KernelFactory
from cobra.core.losses.base import BaseLoss, LossFactory
from cobra.core.optimizers.base import BaseOptimizer
from cobra.core.optimizers.gradient.base import BaseGradientOptimizer, GradientOptimizerFactory
from cobra.core.optimizers.search.base import BaseSearchOptimizer, SearchOptimizerFactory
from cobra.core.spaces.base import SpaceNormalizerFactory
from cobra.core.splitters.base import BaseDataSplitter, SplitterFactory

class MixCOBRARegressor(ABC, SkBaseEstimator, RegressorMixin):
	"""
	MixCOBRARegressor
	"""
	def __init__(
		self,
		estimators: list[str | BaseEstimator] | None = None,
		estimators_params: dict[str, Any] | None = None,
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
		opt_method: str = "grad",
		one_parameter: bool = False,
		random_state: int | None = None
	):
		self.estimators = estimators
		self.estimators_params = estimators_params
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
		self.opt_method = opt_method
		self.one_parameter = one_parameter
		self.random_state = random_state
        
	
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

	def _load_predictions(self, X: np.ndarray):
		cols = []
		for est in self.estimators_:
			preds = est.predict(X)
			cols.append(preds)
		return np.column_stack(cols)

	def _space_normalize(self, X, model_outputs):
		normalizer = SpaceNormalizerFactory.create(
			"mixcobra",
			norm_constant_x=self.norm_constant_x,
			norm_constant_y=self.norm_constant_y
		)
		return normalizer.transform(X, model_outputs)

	def _resolve_component(self):
		self.distance_ : BaseDistance = DistanceFactory.create(
			self.distance,
			**(self.distance_params or {})
		)
		self.kernel_ : BaseKernel = KernelFactory.create(
			self.kernel,
			**(self.kernel_params or {})
		)
		self.aggregator_ : BaseAggregator = AggregatorFactory.create(
			self.aggregator,
			**(self.aggregator_params or {})
		)
		self.loss_ : BaseLoss = LossFactory.create(
			self.loss,
			**(self.loss_params or {})
		)
		self.splitter_ : BaseDataSplitter = SplitterFactory.create(
			'kfold',
			n_splits=5,
			random_state=self.random_state
		)

		self.adapter_ : BaseKernelAdapter = KernelAdapterFactory.create(
			"mixcobra",
			alpha=1.0,
			beta=1.0
		)
	
	def objective_1d(self, params):
		alpha = params[0]
		beta = 0.0
		self.adapter_.set_params(alpha=alpha, beta=beta)
		dist_input = self.distance_.matrix(self.X_l_norm_, self.X_l_norm_)
		dist_output = self.distance_.matrix(self.Y_l_norm_, self.Y_l_norm_)
		mix_distance = np.hstack((dist_input, dist_output))
		D = self.adapter_.transform(mix_distance)
		K = self.kernel_(D)

		preds = np.empty(self.y_l_.shape[0], dtype=float)

		folds = self.splitter_.split(self.X_l_, self.y_l_)

		for train_idx, val_idx in folds:
			w = K[val_idx][:, train_idx]
			y_train = self.y_l_[train_idx]
			for i in range(len(val_idx)):
				if np.allclose(w[i].sum(), 0.0):
					preds[val_idx[i]] = np.mean(y_train)
				else:
					preds[val_idx[i]] = self.aggregator_.aggregate(y_train, w[i])
		
		return self.loss_(self.y_l_, preds)

	def objective_2d(self, params):
		alpha, beta = params
		self.adapter_.set_params(alpha=alpha, beta=beta)
		dist_input = self.distance_.matrix(self.X_l_norm_, self.X_l_norm_)
		dist_output = self.distance_.matrix(self.Y_l_norm_, self.Y_l_norm_)
		D = self.adapter_.transform(dist_input, dist_output)
		K = self.kernel_(D)

		preds = np.empty(self.y_l_.shape[0], dtype=float)

		folds = self.splitter_.split(self.X_l_, self.y_l_)

		for train_idx, val_idx in folds:
			w = K[val_idx][:, train_idx]
			y_train = self.y_l_[train_idx]
			for i in range(len(val_idx)):
				if np.allclose(w[i].sum(), 0.0):
					preds[val_idx[i]] = np.mean(y_train)
				else:
					preds[val_idx[i]] = self.aggregator_.aggregate(y_train, w[i])
		
		return self.loss_(self.y_l_, preds)


	def _optimize_hyperparameters(self):
		if self.opt_method == "grad":
			self.optimizer_ : BaseGradientOptimizer = GradientOptimizerFactory.create(
				self.optimizer,
				**(self.optimizer_params or {}),
				random_state=self.random_state
			)

			if self.one_parameter:
				params, history = self.optimizer_(self.objective_1d, np.array([1.0]))
			else:
				params, history = self.optimizer_(self.objective_2d, np.array([1.0, 1.0]))
		else:
			self.optimizer_ : BaseSearchOptimizer = SearchOptimizerFactory.create(
				self.optimizer,
				**(self.optimizer_params or {}),
				random_state=self.random_state
			)

			param_grid = {}
			if self.one_parameter:
				param_grid["alpha"] = self.alpha_list if self.alpha_list is not None else np.linspace(0, 2, 10)
				params, history = self.optimizer_(self.objective_1d, param_grid)
			else:
				param_grid["alpha"] = self.alpha_list if self.alpha_list is not None else np.linspace(0, 2, 10)
				param_grid["beta"] = self.beta_list if self.beta_list is not None else np.linspace(0, 2, 10)
				params, history = self.optimizer_(self.objective_2d, param_grid)
		
		# store optimization outputs
		self.optimization_outputs_ = {
			"method": self.opt_method,
			"params": params,
			"history": history
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

		# fit base estimators on training set
		self.estimators_ = self._fit_estimators(self.X_k_, self.y_k_)

		# load predictions for aggregation set
		if not self.as_predictions_:
			model_outputs = self._load_predictions(self.X_l_)
		else:
			model_outputs = self.X_l_
		
		# normalize space
		self.X_l_norm_, self.Y_l_norm_ = self._space_normalize(self.X_l_, model_outputs)

		# resolve components
		self._resolve_component()

		# optimize hyperparameters
		self._optimize_hyperparameters()
		return self
		
	def predict(
		self,
		X: np.ndarray,
		pred_X: np.ndarray | None = None,
		alpha: float | None = None,
		beta: float | None = None,
		bandwidth: float | None = None
	) -> np.ndarray:
		check_is_fitted(self)
		X = check_array(X)

		if pred_X is None:
			pred_X = X

		preds = self._load_predictions(X)
		X_norm, Y_norm = self._space_normalize(pred_X, preds)

		dist_input = self.distance_.matrix(X_norm, self.X_l_norm_)
		dist_output = self.distance_.matrix(Y_norm, self.Y_l_norm_)

		if self.one_parameter:
			alpha = self.optimization_outputs_["params"][0]
			beta = 0.0
			mix_distance = np.hstack((dist_input, dist_output))
			self.adapter_.set_params(alpha=alpha, beta=beta)
			D = self.adapter_.transform(mix_distance)
			K = self.kernel_(D)
		else:
			alpha, beta = self.optimization_outputs_["params"]
			self.adapter_.set_params(alpha=alpha, beta=beta)
			D = self.adapter_.transform(dist_input, dist_output)
			K = self.kernel_(D)
		
		W = self.kernel_(D)
		outputs = np.empty(W.shape[0], dtype=float)

		for i in range(W.shape[0]):
			w = W[i]
			if np.allclose(w.sum(), 0.0):
				outputs[i] = np.mean(self.y_l_)
			else:
				outputs[i] = self.aggregator_.aggregate(self.y_l_, w)

		return outputs
