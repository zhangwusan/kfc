"""
Combine Classifier
-------------------
Pipeline:
    Estimators → Prediction Vectors → Exact Matching → Majority Vote
"""

from typing import List, Union, Dict

import numpy as np
from sklearn.base import BaseEstimator as SklearnBaseEstimator

from cobra.core.prediction_vector import PredictionVectorizer
from cobra.estimators.base import BaseEstimator

from cobra.utils.resolve import (
    resolve_from_aggregator,
    resolve_from_distance,
    resolve_from_estimators,
    resolve_from_kernel
)


class CombineClassifier(SklearnBaseEstimator):
    """
    Combine Classifier (COBRA-style hard voting model)

    This model implements:
        - prediction-space representation via base estimators
        - exact matching in prediction space
        - majority vote aggregation
    """

    def __init__(
        self,
        estimators: List[Union[BaseEstimator, str]],
        distance: str = "exact",
        kernel: str = "indicator",
        aggregator: str = "majority_vote",

        estimators_params: dict = None,
        distance_params: dict = None,
        kernel_params: dict = None,
        aggregator_params: dict = None,
    ):
        self.estimators = estimators
        self.distance = distance
        self.kernel = kernel
        self.aggregator = aggregator

        self.estimators_params = estimators_params or {}
        self.distance_params = distance_params or {}
        self.kernel_params = kernel_params or {}
        self.aggregator_params = aggregator_params or {}

        self.is_fitted = False

    # ------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit base estimators and build prediction space.
        """

        # resolve estimators
        self.estimators_: Dict[str, BaseEstimator] = resolve_from_estimators(
            self.estimators,
            self.estimators_params
        )

        # fit estimators
        for est in self.estimators_.values():
            est.fit(X, y)

        # prediction vectorizer
        self.vectorizer_ = PredictionVectorizer(list(self.estimators_.values()))

        # reference data
        self.X_ref_ = X
        self.y_ref_ = y

        # prediction vectors (CRITICAL CACHING STEP)
        self.R_ref_ = self.vectorizer_.transform(X)

        # resolve pipeline components
        self.distance_ = resolve_from_distance(self.distance, self.distance_params)
        self.kernel_ = resolve_from_kernel(self.kernel, self.kernel_params)
        self.aggregator_ = resolve_from_aggregator(self.aggregator, self.aggregator_params)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input X.
        """

        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")

        predictions = []

        for x in X:
            r_x = self.vectorizer_.transform_single(x)

            # STEP 1: exact matching in prediction space
            mask = np.array([
                self.distance_.compute(r_i, r_x)
                for r_i in self.R_ref_
            ])

            idx = np.where(mask)[0]

            # STEP 2: retrieve labels
            y_subset = self.y_ref_[idx] if len(idx) > 0 else np.array([])

            # STEP 3: aggregation
            pred = self.aggregator_.aggregate(
                weights=np.ones(len(y_subset)),
                y=y_subset
            )

            # fallback
            if pred is None:
                pred = self._global_majority()

            predictions.append(pred)

        return np.array(predictions)

    # ------------------------------------------------------------
    # FALLBACK RULE
    # ------------------------------------------------------------
    def _global_majority(self):
        values, counts = np.unique(self.y_ref_, return_counts=True)
        return values[np.argmax(counts)]
