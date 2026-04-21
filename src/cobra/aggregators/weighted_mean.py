
from __future__ import annotations
import numpy as np

from cobra.aggregators.base import BaseAggregator
from cobra.factories.aggregator import AggregatorFactory

@AggregatorFactory.register("weighted_mean", "mixcobra_mean")
class WeightedMeanAggregator(BaseAggregator):
    """
    Continuous weighted averaging.

    Used in MixCOBRA regression.
    """

    def aggregate(self, weights: np.ndarray, y: np.ndarray):
        if len(y) == 0:
            return 0.0

        weights = weights / (np.sum(weights) + 1e-8)
        return np.sum(weights * y)
