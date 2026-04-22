"""Concrete distance metrics used in consensus-space comparisons."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import BaseDistance, DistanceFactory


def _as_2d(arr: ArrayLike) -> np.ndarray:
    """Convert input to a 2D float array."""
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    if out.ndim != 2:
        raise ValueError("Expected a 1D or 2D array-like input.")
    return out


@DistanceFactory.register("euclidean", "l2")
class EuclideanDistance(BaseDistance):
    """Standard Euclidean ($L_2$) distance."""

    def pairwise(self, query: ArrayLike, candidates: ArrayLike) -> np.ndarray:
        q = _as_2d(query)
        c = _as_2d(candidates)
        if q.shape[1] != c.shape[1]:
            raise ValueError("query and candidates must have the same feature dimension.")
        return np.linalg.norm(c - q[0], axis=1)


@DistanceFactory.register("manhattan", "l1")
class ManhattanDistance(BaseDistance):
    """Manhattan ($L_1$) distance, robust to single-coordinate outliers."""

    def pairwise(self, query: ArrayLike, candidates: ArrayLike) -> np.ndarray:
        q = _as_2d(query)
        c = _as_2d(candidates)
        if q.shape[1] != c.shape[1]:
            raise ValueError("query and candidates must have the same feature dimension.")
        return np.sum(np.abs(c - q[0]), axis=1)


@DistanceFactory.register("hamming")
class HammingDistance(BaseDistance):
    """Fraction of mismatched coordinates, useful for discretized outputs."""

    def pairwise(self, query: ArrayLike, candidates: ArrayLike) -> np.ndarray:
        q = _as_2d(query)
        c = _as_2d(candidates)
        if q.shape[1] != c.shape[1]:
            raise ValueError("query and candidates must have the same feature dimension.")
        return np.mean(c != q[0], axis=1).astype(float)
