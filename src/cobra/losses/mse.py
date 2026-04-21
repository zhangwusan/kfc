
from __future__ import annotations
import numpy as np

from cobra.losses.base import BaseLoss, LossFactory

@LossFactory.register("mse", "mean_squared_error")
class MSELoss(BaseLoss):
    def __call__(self, y_true, y_pred, weight=None, **kwargs):
        return np.mean((y_true - y_pred) ** 2)
