"""Consensus-space projectors for COBRA and MIXCOBRA style pipelines."""

from .base import BaseSpaceProjector
from .base import SpaceProjectorFactory
from .builtin import DiscreteProjector, PredictionProjector, TradeOffProjector

__all__ = [
	"BaseSpaceProjector",
	"SpaceProjectorFactory",
	"DiscreteProjector",
	"TradeOffProjector",
	"PredictionProjector",
]