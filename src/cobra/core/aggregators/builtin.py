"""Concrete aggregator implementations for COBRA-style consensus."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import AggregatorFactory, BaseAggregator, _as_1d


@AggregatorFactory.register("mean", "simple_mean")
class SimpleMeanAggregator(BaseAggregator):
    """Return the arithmetic mean of candidate values."""

    def aggregate(self, values: ArrayLike, weights: ArrayLike | None = None) -> float:
        vals = _as_1d(values)
        return float(np.mean(vals))


@AggregatorFactory.register("weighted_mean", "wmean")
class WeightedMeanAggregator(BaseAggregator):
    """Return a weighted mean, with safe fallback to unweighted mean."""

    def aggregate(self, values: ArrayLike, weights: ArrayLike | None = None) -> float:
        vals = _as_1d(values)
        if weights is None:
            return float(np.mean(vals))

        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != vals.size:
            raise ValueError("weights and values must have the same length.")

        w_sum = np.sum(w)
        if np.isclose(w_sum, 0.0):
            return float(np.mean(vals))
        return float(np.sum(vals * w) / w_sum)


@AggregatorFactory.register("majority_vote", "vote")
class MajorityVoteAggregator(BaseAggregator):
    """Return the most frequent class label among candidate values."""

    def aggregate(self, values: ArrayLike, weights: ArrayLike | None = None) -> float:
        vals = np.asarray(values).reshape(-1)
        if vals.size == 0:
            raise ValueError("Cannot aggregate an empty set of values.")
        uniq, counts = np.unique(vals, return_counts=True)
        return float(uniq[np.argmax(counts)])
