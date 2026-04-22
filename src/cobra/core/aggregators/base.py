"""Base interfaces for aggregation strategies.

Aggregators convert neighbor targets (and optional weights) into a single
prediction used as the final consensus output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseAggregator(ABC):
    """Abstract interface for all aggregation rules."""

    @abstractmethod
    def aggregate(self, values: ArrayLike, weights: ArrayLike | None = None) -> float:
        """Aggregate a set of values into a single scalar prediction."""
        raise NotImplementedError


def _as_1d(values: ArrayLike) -> np.ndarray:
    """Normalize values to a 1D float array for stable aggregation."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot aggregate an empty set of values.")
    return arr


class AggregatorFactory(BaseFactory):
    """Registry-backed factory for aggregation strategies."""