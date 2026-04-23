"""Concrete distance metrics used in consensus-space comparisons."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import BaseDistance, DistanceFactory

@DistanceFactory.register("euclidean", "l2", "lp")
class EuclideanDistance(BaseDistance):
    # -------------------------
    # Pairwise
    # -------------------------
    def pairwise(self, query, candidates, p: int = 2):
        q = np.asarray(query, dtype=float)
        X = np.asarray(candidates, dtype=float)

        if q.ndim == 1:
            q = q[None, :]

        diff = np.abs(X - q)
        return np.sum(diff ** p, axis=1) ** (1 / p)
    # -------------------------
    # Matrix
    # -------------------------
    def matrix(self, X, p: int = 2):
        X = np.asarray(X, dtype=float)

        diff = np.abs(X[:, None, :] - X[None, :, :])
        return np.sum(diff ** p, axis=-1) ** (1 / p)

@DistanceFactory.register("manhattan", "l1")
class ManhattanDistance(BaseDistance):
    """
    Manhattan (L1) distance implementation.

    -------------------------
    Supports:
    -------------------------
    - pairwise(query, X) → (n,)
    - matrix(X) → (n, n)
    - tensor(X_list) → (k, n, n)
    """

    # -------------------------
    # Pairwise
    # -------------------------
    def pairwise(self, query: ArrayLike, candidates: ArrayLike, p: int = 1) -> np.ndarray:
        query = np.asarray(query, dtype=float)
        X = np.asarray(candidates, dtype=float)

        if query.ndim == 1:
            query = query[None, :]

        return np.sum(np.abs(X - query), axis=1)

    # -------------------------
    # Matrix
    # -------------------------
    def matrix(self, X: ArrayLike, p: int = 1) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        diff = np.abs(X[:, None, :] - X[None, :, :])
        return np.sum(diff, axis=-1)

@DistanceFactory.register("hamming")
class HammingDistance(BaseDistance):
    """
    Hamming distance:
        proportion of mismatched dimensions
    """

    def pairwise(self, query: ArrayLike, candidates: ArrayLike) -> np.ndarray:
        q = np.asarray(query)
        X = np.asarray(candidates)

        return np.mean(X != q, axis=1)

    def matrix(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X)

        # (n,1,d) != (1,n,d)
        diff = X[:, None, :] != X[None, :, :]
        return np.mean(diff, axis=-1)
