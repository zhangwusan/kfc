"""Aggregation module for final COBRA consensus outputs."""

from .base import BaseAggregator
from .base import AggregatorFactory
from .builtin import MajorityVoteAggregator, SimpleMeanAggregator, WeightedMeanAggregator

__all__ = [
	"BaseAggregator",
	"AggregatorFactory",
	"MajorityVoteAggregator",
	"SimpleMeanAggregator",
	"WeightedMeanAggregator",
]