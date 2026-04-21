
from __future__ import annotations
import numpy as np

from gradientcobra.distances.base import BaseDistance
from gradientcobra.factories.distance import DistanceFactory


@DistanceFactory.register("exact", "cobra_match")
class ExactMatchDistance(BaseDistance):
    """
    Binary identity check used in COBRA.

    Returns True only if ALL predictions match exactly.
    """

    def compute(self, r_i: np.ndarray, r_x: np.ndarray) -> bool:
        return np.array_equal(r_i, r_x)
