"""
CombineClassifier
"""

from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.base import BaseEstimator as SkBaseEstimator, clone
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from cobra.core.aggregators.base import AggregatorFactory, BaseAggregator
from cobra.core.distances.base import BaseDistance, DistanceFactory
from cobra.core.estimators.base import BaseEstimator, EstimatorFactory
from cobra.core.kernels.base import BaseKernel, KernelFactory
from cobra.core.spaces.base import BaseSpaceProjector, SpaceProjectorFactory
from cobra.core.splitters.base import SplitterFactory


class CombineClassifier(ABC, SkBaseEstimator):

    def __init__(
        self,
        estimators: List[Union[str, BaseEstimator]] | None = None,
        estimators_params: Dict[str, Any] | None = None,
        splitter: str = "holdout",
        splitter_params: Dict[str, Any] | None = None,
        distance: str = "hamming",
        distance_params: Dict[str, Any] | None = None,
        kernel: str = "indicator",
        kernel_params: Dict[str, Any] | None = None,
        aggregator: str = "majority_vote",
        aggregator_params: Dict[str, Any] | None = None,
        random_state: int | None = None,
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
            splitter = SplitterFactory.create(
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
            "logistic_regression",
            "random_forest",
            "svm",
            "knn",
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
    
    def _space_projector(self, X, pred_matrix):
        projector : BaseSpaceProjector = SpaceProjectorFactory.create("combine_classifier")
        return projector.transform(X, pred_matrix)

    def fit(
        self,
        X : np.ndarray,
        y : np.ndarray,
        X_l: np.ndarray | None = None,
        y_l: np.ndarray | None = None,
    ):
        
        # Resolve fit context
        (
            self.X_k_, self.y_k_,
            self.X_l_, self.y_l_,
            self.iloc_k_, self.iloc_l_,
        ) = self._resolve_fit_split_context(X, y, X_l, y_l)

        self.classes_ = np.unique(self.y_k_)

        if not self.as_predictions_:
            self.base_estimators_ = self._fit_estimators(self.X_k_, self.y_k_)
            pred_l = self._prediction_matrix(self.X_l_)
            self.z_l_ = self._space_projector(self.X_l_, pred_l)
        else:
            self.z_l_ = self.X_l_
        
        # create distance, kernel, aggregator
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

        classes, counts = np.unique(self.y_k_, return_counts=True)
        self.global_majority_class_ = classes[np.argmax(counts)]

        return self

    def predict(self, X):
        check_is_fitted(self, ["z_l_", "distance_", "kernel_", "aggregator_"])

        X = check_array(X)

        pred_x = self._prediction_matrix(X)
        z_x = self._space_projector(X, pred_x)

        outputs = []

        for row in z_x:
            d = self.distance_.pairwise(row, self.z_l_)
            w = self.kernel_(d)

            mask = w > 0

            if not np.any(mask):
                outputs.append(self.global_majority_class_)
                continue

            y_sub = self.y_l_[mask]
            w_sub = w[mask]

            outputs.append(
                self.aggregator_.aggregate(y_sub, w_sub)
            )

        return np.asarray(outputs)
