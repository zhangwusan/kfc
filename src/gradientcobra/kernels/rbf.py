import numpy as np

from gradientcobra.factories.kernel import KernelFactory
from gradientcobra.kernels.base import BaseKernel


@KernelFactory.register("rbf", "gaussian")
class RBFKernel(BaseKernel):
    """
    Smooth Gaussian kernel:

        K(d) = exp(-||d||^2 / (2h^2))
    """

    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def __call__(self, d: float) -> float:
        return np.exp(-(d ** 2) / (2 * self.bandwidth ** 2))
