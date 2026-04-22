from __future__ import annotations

from typing import Any
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from cobra.core import (
    AggregatorFactory,
    DistanceFactory,
    KernelFactory,
)
from cobra.core.optimizers.base import OptimizerFactory
from cobra.core.spaces.base import SpaceProjectorFactory
from cobra.utils.resolve import (
    resolve_from_estimators,
    resolve_from_splitter,
    resolve_from_distance,
    resolve_from_aggregator,
    resolve_from_loss,
    resolve_from_optimizer,
)


class GradientCOBRA(BaseEstimator, RegressorMixin):
    """
    GradientCOBRA with explicit projection space.

    Key idea:
        Z = projector(X, prediction_matrix(X))

    All geometry is defined in Z-space.
    """

    def __init__(
        self,
        estimators: list[Any] | None = None,
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
        optimizer: str = "grid",
        optimizer_params: dict[str, Any] | None = None,
        projector: str = "prediction_only",
        projector_params: dict[str, Any] | None = None,
        bandwidth_grid: np.ndarray | None = None,
        initial_bandwidth: float = 1.0,
        random_state: int | None = None,
    ):
        self.estimators = estimators
        self.estimators_params = estimators_params or {}

        self.splitter = splitter
        self.splitter_params = splitter_params or {}

        self.distance = distance
        self.distance_params = distance_params or {}

        self.kernel = kernel
        self.kernel_params = kernel_params or {}

        self.aggregator = aggregator
        self.aggregator_params = aggregator_params or {}

        self.loss = loss
        self.loss_params = loss_params or {}

        self.optimizer = optimizer
        self.optimizer_params = optimizer_params or {}

        self.projector = projector
        self.projector_params = projector_params or {}

        self.bandwidth_grid = bandwidth_grid
        self.initial_bandwidth = float(initial_bandwidth)

        self.random_state = random_state

    # ------------------------------------------------------------
    # Prediction matrix
    # ------------------------------------------------------------
    def _predict_matrix(self, X: np.ndarray, estimators: list[Any]) -> np.ndarray:
        return np.column_stack([
            est.predict(X).astype(float)
            for est in estimators
        ])

    # ------------------------------------------------------------
    # Projector space
    # ------------------------------------------------------------
    def _project(self, X: np.ndarray, pred: np.ndarray) -> np.ndarray:
        projector = SpaceProjectorFactory.create(
            self.projector,
            **self.projector_params
        )
        return projector.transform(X, pred)

    # ------------------------------------------------------------
    # Kernel resolution
    # ------------------------------------------------------------
    def _resolve_kernel(self, bandwidth: float):
        params = dict(self.kernel_params)

        if self.kernel in {"indicator", "hard"}:
            params["epsilon"] = float(bandwidth)
        else:
            params["bandwidth"] = float(bandwidth)

        return KernelFactory.create(self.kernel, **params)

    # ------------------------------------------------------------
    # Leave-one-out error
    # ------------------------------------------------------------
    def _leave_one_out_error(self, bandwidth: float) -> float:
        kernel = self._resolve_kernel(bandwidth)

        n = self.z_agg_.shape[0]
        preds = np.empty(n, dtype=float)

        for i in range(n):
            d = self.distance_.pairwise(self.z_agg_[i], self.z_agg_)
            w = np.asarray(kernel(d), dtype=float)

            w[i] = 0.0

            if np.allclose(w.sum(), 0.0):
                preds[i] = float(np.mean(self.y_agg_))
            else:
                preds[i] = float(self.aggregator_.aggregate(self.y_agg_, w))

        return float(self.loss_(self.y_agg_, preds))

    # ------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------
    def _optimize_bandwidth(self) -> float:
        if self.optimizer == "grid" and self.bandwidth_grid is not None:
            grid = np.asarray(self.bandwidth_grid, dtype=float)
            scores = np.array([self._leave_one_out_error(v) for v in grid])

            best = float(grid[np.argmin(scores)])

            self.optimization_outputs_ = {
                "method": "grid",
                "bandwidth": best,
                "risk": float(np.min(scores)),
            }
            return best

        optimizer = OptimizerFactory.create(
            self.optimizer,
            **self.optimizer_params
        )

        best = float(
            optimizer.optimize(self._leave_one_out_error, self.initial_bandwidth)
        )

        self.optimization_outputs_ = {
            "method": self.optimizer,
            "bandwidth": best,
            "risk": float(self._leave_one_out_error(best)),
        }

        return best

    # ------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientCOBRA":
        X, y = check_X_y(X, y, y_numeric=True)
        y = np.asarray(y, dtype=float)

        # ---- split ----
        splitter = resolve_from_splitter(
            self.splitter,
            self.splitter_params,
        )
        idx_train, idx_agg = splitter.split(X, y)

        self.x_train_, self.y_train_ = X[idx_train], y[idx_train]
        self.x_agg_, self.y_agg_ = X[idx_agg], y[idx_agg]

        # ---- models ----
        self.base_estimators_ = resolve_from_estimators(
            self.estimators,
            self.estimators_params,
            default_estimators=[
                "linear",
                "ridge",
                "lasso",
                "knn",
                "random_forest",
                "svm",
            ],
        )

        for est in self.base_estimators_:
            est.fit(self.x_train_, self.y_train_)

        # ---- prediction matrix ----
        raw_pred = self._predict_matrix(self.x_agg_, self.base_estimators_)

        # ---- PROJECTOR SPACE (NEW CORE FIX) ----
        self.z_agg_ = self._project(self.x_agg_, raw_pred)

        # ---- components ----
        self.distance_ = DistanceFactory.create(
            self.distance,
            **self.distance_params or {}
        )

        self.aggregator_ = AggregatorFactory.create(
            self.aggregator,
            **self.aggregator_params or {}
        )

        self.loss_ = resolve_from_loss(
            self.loss,
            self.loss_params,
        )

        # ---- optimize ----
        self.opt_bandwidth_ = self._optimize_bandwidth()
        self.kernel_ = self._resolve_kernel(self.opt_bandwidth_)

        return self

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["z_agg_", "distance_", "kernel_", "aggregator_"])

        X = check_array(X)

        raw_pred = self._predict_matrix(X, self.base_estimators_)
        z_x = self._project(X, raw_pred)

        outputs = np.empty(z_x.shape[0], dtype=float)

        kernel = self._resolve_kernel(self.opt_bandwidth_)

        for i, row in enumerate(z_x):
            d = self.distance_.pairwise(row, self.z_agg_)
            w = np.asarray(kernel(d), dtype=float)

            outputs[i] = float(
                self.aggregator_.aggregate(self.y_agg_, w)
            )

        return outputs
