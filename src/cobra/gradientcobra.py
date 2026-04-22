from __future__ import annotations

from typing import Any
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from cobra.core.optimizers.base import OptimizerFactory
from cobra.utils.resolve import (
    resolve_from_aggregator,
    resolve_from_distance,
    resolve_from_estimators,
    resolve_from_kernel,
    resolve_from_loss,
    resolve_from_optimizer,
    resolve_from_splitter,
)


class GradientCOBRA(BaseEstimator, RegressorMixin):
    """
    GradientCOBRA: Prediction-space aggregation with bandwidth optimization.

    This model implements a generalized COBRA framework:

        1. Split dataset into training and aggregation sets
        2. Train base regressors on training set
        3. Project samples into prediction space
        4. Compute distances between prediction vectors
        5. Apply kernel weighting
        6. Aggregate targets
        7. Optimize kernel bandwidth via leave-one-out risk

    Parameters
    ----------
    estimators : list[str] or None
        List of estimator aliases. If None, defaults are used.

    estimators_params : dict
        Parameters passed to estimator constructors.

    splitter : str
        Data splitting strategy.

    splitter_params : dict
        Parameters for splitter.

    distance : str
        Distance metric in prediction space.

    kernel : str
        Kernel function name.

    aggregator : str
        Aggregation method.

    loss : str
        Loss function used for bandwidth optimization.

    optimizer : str
        Optimization method ("grid", "gradient", etc.)

    bandwidth_grid : np.ndarray or None
        Grid for grid search.

    initial_bandwidth : float
        Initial value for gradient-based optimization.

    random_state : int or None
        Random seed.
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

        self.bandwidth_grid = bandwidth_grid
        self.initial_bandwidth = float(initial_bandwidth)

        self.random_state = random_state

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _predict_matrix(self, X: np.ndarray, estimators: list[Any]) -> np.ndarray:
        """Project input X into prediction space."""
        return np.column_stack([est.predict(X).astype(float) for est in estimators])

    def _resolve_kernel(self, bandwidth: float):
        """
        Resolve kernel with correct parameter mapping.

        Handles difference between:
            - soft kernels → bandwidth
            - hard kernels → epsilon
        """
        params = dict(self.kernel_params)

        if self.kernel in {"indicator", "hard"}:
            params["epsilon"] = float(bandwidth)
        else:
            params["bandwidth"] = float(bandwidth)

        return resolve_from_kernel(self.kernel, params)

    # ------------------------------------------------------------
    # Objective for bandwidth optimization
    # ------------------------------------------------------------

    def _leave_one_out_error(self, bandwidth: float) -> float:
        kernel = self._resolve_kernel(bandwidth)

        n = self.pred_agg_.shape[0]
        preds = np.empty(n, dtype=float)

        for i in range(n):
            d = self.distance_.pairwise(self.pred_agg_[i], self.pred_agg_)
            w = np.asarray(kernel(d), dtype=float)

            w[i] = 0.0  # leave-one-out

            if np.allclose(w.sum(), 0.0):
                preds[i] = float(np.mean(self.y_agg_))
            else:
                preds[i] = float(self.aggregator_.aggregate(self.y_agg_, w))

        return float(self.loss_(self.y_agg_, preds))

    # ------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------

    def _optimize_bandwidth(self) -> float:
        """Select optimal bandwidth."""

        # ---- Grid search ----
        if self.optimizer == "grid" and self.bandwidth_grid is not None:
            grid = np.asarray(self.bandwidth_grid, dtype=float).ravel()
            scores = np.array([self._leave_one_out_error(v) for v in grid])

            best = float(grid[int(np.argmin(scores))])

            self.optimization_outputs_ = {
                "method": "grid",
                "bandwidth": best,
                "risk": float(np.min(scores)),
            }
            return best

        # ---- General optimizer ----
        optimizer = resolve_from_optimizer(self.optimizer, self.optimizer_params)

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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_l: np.ndarray | None = None,
        y_l: np.ndarray | None = None,
        as_predictions: bool = False,
    ) -> "GradientCOBRA":

        X, y = check_X_y(X, y, y_numeric=True)
        y = np.asarray(y, dtype=float)

        self.as_predictions_ = bool(as_predictions)

        # ---- Data splitting ----
        if X_l is not None or y_l is not None:
            if X_l is None or y_l is None:
                raise ValueError("Both X_l and y_l must be provided.")

            X_l, y_l = check_X_y(X_l, y_l, y_numeric=True)

            self.x_train_, self.y_train_ = X, y
            self.x_agg_, self.y_agg_ = X_l, y_l
            self.as_predictions_ = False
            
        elif self.as_predictions_:
            self.x_train_, self.y_train_ = None, None
            self.x_agg_, self.y_agg_ = None, y
            self.pred_agg_ = X.astype(float)
            self.as_predictions_ = True
            
        else:
            splitter = resolve_from_splitter(
                self.splitter,
                self.splitter_params,
            )
            idx_train, idx_agg = splitter.split(X, y)
            self.x_train_, self.y_train_ = X[idx_train], y[idx_train]
            self.x_agg_, self.y_agg_ = X[idx_agg], y[idx_agg]
            self.as_predictions_ = False

        # ---- Train estimators ----
        if not self.as_predictions_:
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

            self.pred_agg_ = self._predict_matrix(
                self.x_agg_,
                self.base_estimators_,
            )

        # ---- Resolve components ----
        self.distance_ = resolve_from_distance(
            self.distance,
            self.distance_params,
        )

        self.aggregator_ = resolve_from_aggregator(
            self.aggregator,
            self.aggregator_params,
        )

        self.loss_ = resolve_from_loss(
            self.loss,
            self.loss_params,
        )

        # ---- Optimize bandwidth ----
        self.opt_bandwidth_ = self._optimize_bandwidth()

        self.kernel_ = self._resolve_kernel(self.opt_bandwidth_)

        return self

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        bandwidth: float | None = None,
    ) -> np.ndarray:

        check_is_fitted(
            self,
            ["pred_agg_", "distance_", "aggregator_", "opt_bandwidth_"],
        )

        X = check_array(X)

        pred_x = (
            X.astype(float)
            if self.as_predictions_
            else self._predict_matrix(X, self.base_estimators_)
        )

        kernel = self._resolve_kernel(
            self.opt_bandwidth_ if bandwidth is None else float(bandwidth)
        )

        outputs = np.empty(pred_x.shape[0], dtype=float)

        for i, row in enumerate(pred_x):
            d = self.distance_.pairwise(row, self.pred_agg_)
            w = np.asarray(kernel(d), dtype=float)
            outputs[i] = float(
                self.aggregator_.aggregate(self.y_agg_, w)
            )

        return outputs
