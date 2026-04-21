from __future__ import annotations
import numpy as np

from cobra.factories.kernel import KernelFactory
from cobra.kernels.base import BaseKernel

@KernelFactory.register("indicator", "cobra_kernel")
class IndicatorKernel(BaseKernel):
    """
    Hard binary kernel used in COBRA.

    K(d) = 1 if match else 0
    """

    def __call__(self, match: bool) -> float:
        return 1.0 if match else 0.0
