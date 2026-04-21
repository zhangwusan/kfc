
from __future__ import annotations
import numpy as np

from cobra.factories.kernel import KernelFactory
from cobra.kernels.base import BaseKernel


@KernelFactory.register("polynomial")
class PolynomialKernel(BaseKernel):
    """
    Polynomial similarity kernel.
    """

    def __init__(self, degree=2, coef0=1.0):
        self.degree = degree
        self.coef0 = coef0

    def __call__(self, d: float) -> float:
        return (self.coef0 + d) ** self.degree
