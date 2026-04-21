from __future__ import annotations
import numpy as np
from cobra.aggregators.base import BaseAggregator
from cobra.factories.aggregator import AggregatorFactory

@AggregatorFactory.register("kernel_regression", "gradientcobra")
class KernelRegressionAggregator(BaseAggregator):
    """
    Standard Nadaraya-Watson kernel regression.

    Final rule:

        f(x) = Σ w_i y_i / Σ w_i
    """

    def aggregate(self, weights: np.ndarray, y: np.ndarray):
        denom = np.sum(weights) + 1e-8
        return np.sum(weights * y) / denom
