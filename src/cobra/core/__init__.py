"""Core modular primitives for the COBRA library."""

from .aggregators import AggregatorFactory
from .distances import DistanceFactory
from .estimators import EstimatorFactory
from .kernels import KernelFactory
from .losses import LossFactory
from .optimizers import OptimizerFactory
from .spaces import SpaceProjectorFactory
from .splitters import SplitterFactory

__all__ = [
	"EstimatorFactory",
	"DistanceFactory",
	"KernelFactory",
	"AggregatorFactory",
	"LossFactory",
	"OptimizerFactory",
	"SpaceProjectorFactory",
	"SplitterFactory",
]
