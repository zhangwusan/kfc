
from __future__ import annotations
import numpy as np

from cobra.distances.base import BaseDistance, DistanceFactory

@DistanceFactory.register("joint", "input_output")
class JointDistance(BaseDistance):
    """
    Combines input-space and prediction-space distances.

    Used in MixCOBRA:

        D = (||X_i - x|| / α ,  ||r_i - r_x|| / β)
    """

    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def compute(self, x_i, x, r_i, r_x):
        d_input = np.linalg.norm(x_i - x) / self.alpha
        d_output = np.linalg.norm(r_i - r_x) / self.beta
        return d_input, d_output
