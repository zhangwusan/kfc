"""
CombineClassifier
"""

from __future__ import annotations
from abc import ABC
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.base import BaseEstimator as SkBaseEstimator
from sklearn.utils import check_array

from cobra.core.aggregators.base import AggregatorFactory, BaseAggregator
from cobra.core.distances.base import BaseDistance, DistanceFactory
from cobra.core.estimators.base import BaseEstimator, EstimatorFactory
from cobra.core.kernels.base import BaseKernel, KernelFactory


class CombineClassifier(ABC, SkBaseEstimator):

    def __init__(
        self,
        estimators: List[Union[str, BaseEstimator]] | None = None,
        estimators_params: Dict[str, Any] | None = None,
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
        self.distance = distance
        self.distance_params = distance_params
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.aggregator = aggregator
        self.aggregator_params = aggregator_params
        self.random_state = random_state

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
        cols = []
        for est in self.estimators_:
            preds = est.predict(X)
            cols.append(preds)
        return np.column_stack(cols)
    
    def _resolve_components(self):
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
    

    def fit(self, X : np.ndarray, y : np.ndarray):
        self.classes_ = np.unique(y)
        self.estimators_ = self._fit_estimators(X, y)
        self.y_ = self._prediction_matrix(X)
        classes, counts = np.unique(self.y_, return_counts=True)
        self.global_majority_class_ = classes[np.argmax(counts)]

        self._resolve_components()
        return self

    def predict(self, X):
        X = check_array(X)

        preds = self._prediction_matrix(X)
        outputs = []

        D = self.distance_.matrix(preds, self.y_)
        K = self.kernel_(D)

        for i in range(K.shape[0]):
            w = K[i]
            mask = w > 0
            if not np.any(mask):
                outputs.append(self.global_majority_class_)
                continue

            y_sub = self.y_[mask]
            w_sub = w[mask]

            outputs.append(
                self.aggregator_.aggregate(y_sub, w_sub)
            )

        return np.asarray(outputs)
