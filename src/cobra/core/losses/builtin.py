"""Concrete losses for regression and classification objectives."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import BaseLoss, LossFactory


@LossFactory.register("mse", "mean_squared_error")
class MSELoss(BaseLoss):
    """Mean squared error loss."""

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        return float(np.mean((yt - yp) ** 2))


@LossFactory.register("mae", "mean_absolute_error")
class MAELoss(BaseLoss):
    """Mean absolute error loss."""

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt = np.asarray(y_true, dtype=float).reshape(-1)
        yp = np.asarray(y_pred, dtype=float).reshape(-1)
        return float(np.mean(np.abs(yt - yp)))


@LossFactory.register("zero_one", "classification_error")
class ZeroOneLoss(BaseLoss):
    """$0$-$1$ loss for discrete class predictions."""

    def __call__(self, y_true: ArrayLike, y_pred: ArrayLike) -> float:
        yt = np.asarray(y_true).reshape(-1)
        yp = np.asarray(y_pred).reshape(-1)
        return float(np.mean(yt != yp))
