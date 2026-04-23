

from __future__ import annotations

import numpy as np

from cobra.core.distances.base import BaseDistance, DistanceFactory

@DistanceFactory.register("euclidean", "l2")
class EuclideanDistance(BaseDistance):
    def matrix(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        x2 = np.sum(x ** 2, axis=1, keepdims=True)
        y2 = np.sum(y ** 2, axis=1, keepdims=True).T

        xy = x @ y.T

        return np.sqrt(np.maximum(x2 + y2 - 2 * xy, 0.0))

@DistanceFactory.register("manhattan", "l1")
class ManhattanDistance(BaseDistance):
    def matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        y = np.asarray(y)
        return np.sum(
            np.abs(x[:, None, :] - y[None, :, :]),
            axis=2
        )

@DistanceFactory.register("hamming")
class HammingDistance(BaseDistance):
    def matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        y = np.asarray(y)
        return np.mean(
            x[:, None, :] != y[None, :, :],
            axis=2
        )

@DistanceFactory.register("minkowski", "lp")
class MinkowskiDistance(BaseDistance):

    def __init__(self, p: float = 3, **kwargs):
        super().__init__(p=p, **kwargs)

    def matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        y = np.asarray(y)

        p = self.p
        return np.sum(
            np.abs(x[:, None, :] - y[None, :, :]) ** p,
            axis=2
        ) ** (1 / p)


@DistanceFactory.register("cosine")
class CosineDistance(BaseDistance):

    def matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        y = np.asarray(y)
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True).T
        # cosine similarity
        sim = (x @ y.T) / (x_norm * y_norm + 1e-12)
        # cosine distance
        return 1.0 - sim

