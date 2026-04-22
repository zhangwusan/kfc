"""Consensus-space projectors for COBRA and MIXCOBRA style pipelines."""

from .base import BaseSpaceProjector
from .base import SpaceProjectorFactory
from .builtin import MixCOBRASpaceProjector, PredictionOnlyProjector

__all__ = [
	"BaseSpaceProjector",
	"SpaceProjectorFactory",
	"PredictionOnlyProjector",
	"MixCOBRASpaceProjector",
]