"""
Kernel Factory Module

This factory is responsible for managing all kernel implementations
used in the GradientCOBRA / MixCOBRA / COBRA framework.

It follows a registry-based plugin architecture inherited from BaseFactory,
allowing researchers to dynamically register and instantiate kernels
via string identifiers.

This design enables:
- Plug-and-play model swapping
- Reproducible experimental pipelines
- Clean separation between algorithm logic and model implementations
"""

from __future__ import annotations
from typing import Dict

from gradientcobra.factories.base import BaseFactory
from gradientcobra.kernels.base import BaseKernel


class KernelFactory(BaseFactory):
    """
    Factory for managing kernel components.

    This class inherits all functionality from BaseFactory:
    - register() decorator for adding kernels
    - create() for instantiating kernels by name
    - available() for listing registered kernels
    """

    _registry: Dict[str, BaseKernel] = {}
