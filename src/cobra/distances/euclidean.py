
from __future__ import annotations
import numpy as np

from cobra.distances.base import BaseDistance, DistanceFactory
@DistanceFactory.register("euclidean")
class EuclideanDistance(BaseDistance):
    """
    Standard Euclidean distance in input space.
    """

    def compute(self, x_i: np.ndarray, x: np.ndarray) -> float:
        return np.linalg.norm(x_i - x)
