

from __future__ import annotations

import numpy as np

from cobra.core.distances.base import BaseDistance

@BaseDistance.register("euclidean", "l2")
class EuclideanDistance(BaseDistance):
    def __call__(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return np.linalg.norm(x - y, axis=-1)

@BaseDistance.register("manhattan", "l1")
class ManhattanDistance(BaseDistance):
    def __call__(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return np.sum(np.abs(x - y), axis=-1)

@BaseDistance.register("minkowski", "lp")
class MinkowskiDistance(BaseDistance):
    def __call__(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        p = self.params.get("p", 3)
        return np.sum(np.abs(x - y) ** p, axis=-1) ** (1/p)

@BaseDistance.register("cosine")
class CosineDistance(BaseDistance):
    def __call__(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        x_norm = np.linalg.norm(x, axis=-1)
        y_norm = np.linalg.norm(y, axis=-1)
        dot_product = np.sum(x * y, axis=-1)
        return 1 - dot_product / (x_norm * y_norm + 1e-10)

