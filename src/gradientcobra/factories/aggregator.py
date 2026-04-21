"""
Aggregator Factory Module

This factory is responsible for managing all aggregation implementations
used in the GradientCOBRA / MixCOBRA / COBRA framework.

It follows a registry-based plugin architecture inherited from BaseFactory,
allowing researchers to dynamically register and instantiate aggregators
via string identifiers.

This design enables:
- Plug-and-play model swapping
- Reproducible experimental pipelines
- Clean separation between algorithm logic and model implementations
"""
from __future__ import annotations
from typing import Dict

from gradientcobra.aggregators.base import BaseAggregator
from gradientcobra.factories.base import BaseFactory


class AggregatorFactory(BaseFactory):
    """
    Factory for managing aggregation components.

    This class inherits all functionality from BaseFactory:
    - register() decorator for adding aggregators
    - create() for instantiating aggregators by name
    - available() for listing registered aggregators
    """

    _registry: Dict[str, BaseAggregator] = {}
