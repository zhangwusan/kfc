from __future__ import annotations
import numpy as np
from cobra.losses.base import BaseLoss, LossFactory

@LossFactory.register("wmse", "weighted_mse")
class WeightedMSELoss(BaseLoss):
    def __call__(self, y_true, y_pred, weight=None, **kwargs):
        if weight is None:
            raise ValueError("weight required for weighted loss")
        return np.sum(weight * (y_true - y_pred) ** 2) / np.sum(weight)
