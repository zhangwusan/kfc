
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from cobra.core import AggregatorFactory, DistanceFactory, KernelFactory, SplitterFactory


@dataclass
class _ClassifierSpec:
    """Container describing classifier aliases and concrete estimators."""

    name: str
    estimator: Any


class CombineClassifier(BaseEstimator, ClassifierMixin):
    """Mojirsheibani-style COBRA classifier with hard consensus matching.

    Pipeline
    --------
    1. Split data into estimator-training and aggregation subsets.
    2. Fit base classifiers on the training subset.
    3. Build prediction vectors on the aggregation subset.
    4. For a new point, keep only exact vector matches (via Hamming + Indicator).
    5. Aggregate kept labels by majority vote.
    """

    def __init__(
        self,
        estimators: list[Any] | None = None,
        splitter: str = "holdout",
        splitter_params: dict[str, Any] | None = None,
        distance: str = "hamming",
        distance_params: dict[str, Any] | None = None,
        kernel: str = "indicator",
        kernel_params: dict[str, Any] | None = None,
        aggregator: str = "majority_vote",
        aggregator_params: dict[str, Any] | None = None,
        random_state: int | None = None,
    ):
        self.estimators = estimators
        self.splitter = splitter
        self.splitter_params = splitter_params
        self.distance = distance
        self.distance_params = distance_params
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.aggregator = aggregator
        self.aggregator_params = aggregator_params
        self.random_state = random_state

    def _default_estimators(self) -> list[_ClassifierSpec]:
        """Provide a diverse default classifier pool for consensus."""
        return [
            _ClassifierSpec("logistic_regression", LogisticRegression(max_iter=2000, random_state=self.random_state)),
            _ClassifierSpec("random_forest", RandomForestClassifier(n_estimators=300, random_state=self.random_state)),
            _ClassifierSpec("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=self.random_state)),
            _ClassifierSpec("knn", KNeighborsClassifier(n_neighbors=7)),
        ]

    def _resolve_estimators(self) -> list[Any]:
        """Resolve estimator list from aliases or sklearn-compatible instances."""
        alias_map = {spec.name: spec.estimator for spec in self._default_estimators()}

        if self.estimators is None:
            return [clone(est) for est in alias_map.values()]

        resolved: list[Any] = []
        for item in self.estimators:
            if isinstance(item, str):
                key = item.lower()
                if key not in alias_map:
                    raise KeyError(
                        f"Unknown estimator alias '{item}'. "
                        f"Available aliases: {sorted(alias_map.keys())}."
                    )
                resolved.append(clone(alias_map[key]))
            else:
                resolved.append(clone(item))
        return resolved

    def _prediction_matrix(self, x: np.ndarray, estimators: list[Any]) -> np.ndarray:
        """Build a matrix with one column per base classifier prediction."""
        cols = [np.asarray(model.predict(x)).reshape(-1, 1) for model in estimators]
        return np.hstack(cols)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CombineClassifier":
        """Fit classifier pool and cache the aggregation subset representation."""
        x, y = check_X_y(x, y)
        self.classes_ = np.unique(y)

        split_params = dict(self.splitter_params or {})
        split_params.setdefault("random_state", self.random_state)
        splitter = SplitterFactory.create(self.splitter, **split_params)
        idx_train, idx_agg = splitter.split(x, y)

        self.x_train_, self.y_train_ = x[idx_train], y[idx_train]
        self.x_agg_, self.y_agg_ = x[idx_agg], y[idx_agg]
        self.global_majority_class_ = self.classes_[np.argmax(np.bincount(np.searchsorted(self.classes_, self.y_agg_)))]

        self.base_estimators_ = self._resolve_estimators()
        for model in self.base_estimators_:
            model.fit(self.x_train_, self.y_train_)

        self.pred_agg_ = self._prediction_matrix(self.x_agg_, self.base_estimators_)

        distance_params = dict(self.distance_params or {})
        kernel_params = dict(self.kernel_params or {})
        aggregator_params = dict(self.aggregator_params or {})

        self.distance_ = DistanceFactory.create(self.distance, **distance_params)
        self.kernel_ = KernelFactory.create(self.kernel, **kernel_params)
        self.aggregator_ = AggregatorFactory.create(self.aggregator, **aggregator_params)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict labels using exact consensus matches in prediction space."""
        check_is_fitted(self, ["base_estimators_", "pred_agg_", "distance_", "kernel_", "aggregator_"])
        x = check_array(x)

        pred_mat = self._prediction_matrix(x, self.base_estimators_)
        outputs: list[Any] = []

        for row in pred_mat:
            distances = self.distance_.pairwise(row.reshape(1, -1), self.pred_agg_)
            weights = np.asarray(self.kernel_(distances), dtype=float)
            mask = weights > 0.0

            if not np.any(mask):
                outputs.append(self.global_majority_class_)
                continue

            y_subset = self.y_agg_[mask]
            w_subset = weights[mask]
            pred = self.aggregator_.aggregate(y_subset, w_subset)
            outputs.append(pred)

        out = np.asarray(outputs)
        return out.astype(self.classes_.dtype, copy=False)
