"""
Distance Factory Module

This factory is responsible for managing all distance implementations
used in the GradientCOBRA / MixCOBRA / COBRA framework.

It follows a registry-based plugin architecture inherited from BaseFactory,
allowing researchers to dynamically register and instantiate distances
via string identifiers.

This design enables:
- Plug-and-play model swapping
- Reproducible experimental pipelines
- Clean separation between algorithm logic and model implementations
"""
from __future__ import annotations
from typing import Dict

from cobra.distances.base import BaseDistance
from cobra.factories.base import BaseFactory


class DistanceFactory(BaseFactory):
    """
    Factory for managing distance components.

    This class inherits all functionality from BaseFactory:
    - register() decorator for adding distances
    - create() for instantiating distances by name
    - available() for listing registered distances
    """

    _registry: Dict[str, BaseDistance] = {}
