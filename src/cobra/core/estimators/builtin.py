"""
Estimator wrappers for COBRA-style expert pools.

This module provides a unified interface over scikit-learn regressors,
allowing them to be used interchangeably inside ensemble frameworks
such as COBRA, GradientCOBRA, and MixCOBRA.

Design goals:
- unify estimator interface
- ensure factory-based instantiation
- remove boilerplate duplication
- allow flexible hyperparameter passing
"""

from __future__ import annotations

from typing import Any, Optional
from numpy.typing import ArrayLike

import numpy as np
from sklearn.base import BaseEstimator as SkBaseEstimator

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .base import BaseEstimator, EstimatorFactory

class SklearnEstimator(BaseEstimator):
    """
    Generic wrapper for any scikit-learn regressor.

    This class standardizes the fit/predict interface so that all
    estimators can be used inside COBRA-style ensembles.

    Parameters
    ----------
    estimator:
        A scikit-learn compatible regressor instance.
    """

    def __init__(self, estimator: SkBaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, x: ArrayLike, y: ArrayLike) -> "SklearnEstimator":
        """
        Fit underlying estimator.

        Parameters
        ----------
        x : ArrayLike
            Training features.
        y : ArrayLike
            Target values.

        Returns
        -------
        self
        """
        self.estimator.fit(x, y)
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        """
        Generate predictions.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        return np.asarray(self.estimator.predict(x), dtype=float)

    def predict_proba(self, x: ArrayLike) -> np.ndarray:
        """
        Generate class probabilities if supported.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(x), dtype=float)
        else:
            raise NotImplementedError(
                f"{self.estimator.__class__.__name__} does not support predict_proba."
            )

@EstimatorFactory.register("mean_regressor", "dummy_mean")
class MeanRegressor(BaseEstimator):
    """
    Baseline model predicting the mean of training targets.

    Useful as a sanity check or weak baseline inside ensemble pools.
    """

    def __init__(self) -> None:
        self.estimator = DummyRegressor(strategy="mean")

    def fit(self, x: ArrayLike, y: ArrayLike) -> "MeanRegressor":
        self.estimator.fit(x, y)
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        return np.asarray(self.estimator.predict(x), dtype=float)
    
@EstimatorFactory.register("linear")
class LinearRegressorEstimator(SklearnEstimator):
    """
    Ordinary Least Squares Linear Regression.

    Captures linear relationships between features and target.
    """

    def __init__(self) -> None:
        super().__init__(LinearRegression())

@EstimatorFactory.register("ridge")
class RidgeRegressorEstimator(SklearnEstimator):
    """
    Ridge regression with L2 regularization.

    Helps stabilize solutions under multicollinearity.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(Ridge(alpha=alpha))

@EstimatorFactory.register("lasso")
class LassoRegressorEstimator(SklearnEstimator):
    """
    Lasso regression with L1 regularization.

    Performs feature selection by driving some coefficients to zero.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(Lasso(alpha=alpha))

@EstimatorFactory.register("knn")
class KNNRegressorEstimator(SklearnEstimator):
    """
    K-Nearest Neighbors regression.

    Non-parametric model based on local similarity.
    """

    def __init__(self, n_neighbors: int = 7) -> None:
        super().__init__(KNeighborsRegressor(n_neighbors=n_neighbors))

@EstimatorFactory.register("random_forest")
class RandomForestRegressorEstimator(SklearnEstimator):
    """
    Random Forest regression.

    Ensemble of decision trees for robust nonlinear modeling.
    """

    def __init__(self, n_estimators: int = 100, random_state: Optional[int] = None) -> None:
        super().__init__(
            RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
            )
        )

@EstimatorFactory.register("svm")
class SVMRegressorEstimator(SklearnEstimator):
    """
    Support Vector Regression with RBF kernel.

    Effective for high-dimensional nonlinear regression tasks.
    """

    def __init__(self, C: float = 5.0, epsilon: float = 0.05) -> None:
        super().__init__(SVR(C=C, epsilon=epsilon, kernel="rbf"))

@EstimatorFactory.register("logistic_regression")
class LogisticRegressionEstimator(SklearnEstimator):
    """
    Logistic Regression for regression tasks.

    Although primarily a classifier, it can be used in regression settings
    by treating the output as a continuous score.
    """

    def __init__(self, max_iter: int = 5000, random_state: Optional[int] = None) -> None:
        super().__init__(
            LogisticRegression(max_iter=max_iter, random_state=random_state)
        )
    
@EstimatorFactory.register("desicion_tree")
class DecisionTreeRegressorEstimator(SklearnEstimator):
    """
    Decision Tree regression.

    Simple tree-based model that captures nonlinear relationships.
    """

    def __init__(self, max_depth: Optional[int] = None, random_state: Optional[int] = None) -> None:
        super().__init__(
            DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        )

@EstimatorFactory.register("gradient_boosting")
class GradientBoostingRegressorEstimator(SklearnEstimator):
    """
    Gradient Boosting regression.

    Ensemble of weak learners (e.g., decision trees) trained sequentially.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
            )
        )