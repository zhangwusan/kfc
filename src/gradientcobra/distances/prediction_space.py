"""
Distance in prediction space only.
Used in GradientCOBRA.
"""
from __future__ import annotations
import numpy as np

from gradientcobra.distances.base import BaseDistance
from gradientcobra.factories.distance import DistanceFactory



@DistanceFactory.register("prediction", "prediction_space")
class PredictionSpaceDistance(BaseDistance):
    """
    Euclidean distance between prediction vectors:
        D(x, x') = || r(x) - r(x') ||
    """

    def compute(self, r_i: np.ndarray, r_x: np.ndarray) -> float:
        return np.linalg.norm(r_i - r_x)
