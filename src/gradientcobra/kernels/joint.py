
from __future__ import annotations
import numpy as np

from gradientcobra.factories.kernel import KernelFactory
from gradientcobra.kernels.base import BaseKernel


@KernelFactory.register("joint", "mixcobra_kernel")
class JointKernel(BaseKernel):
    """
    Kernel over (input distance, output distance).

    K(d_input, d_output)
    """

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, d_input: float, d_output: float) -> float:
        return np.exp(-self.gamma * (d_input ** 2 + d_output ** 2))
