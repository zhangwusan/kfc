"""
Estimator Factory Module

This factory is responsible for managing all estimator implementations
used in the GradientCOBRA / MixCOBRA / COBRA framework.

It follows a registry-based plugin architecture inherited from BaseFactory,
allowing researchers to dynamically register and instantiate estimators
via string identifiers.

This design enables:
- Plug-and-play model swapping
- Reproducible experimental pipelines
- Clean separation between algorithm logic and model implementations
"""

from __future__ import annotations
from typing import Dict

from gradientcobra.estimators.base import BaseEstimator
from gradientcobra.factories.base import BaseFactory


class EstimatorFactory(BaseFactory):
    """
    Factory for managing estimator components.

    This class inherits all functionality from BaseFactory:
    - register() decorator for adding estimators
    - create() for instantiating estimators by name
    - available() for listing registered estimators
    """

    _registry: Dict[str, BaseEstimator] = {}
