"""Base interface for projecting data into consensus spaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseSpaceProjector(ABC):
    """Map raw inputs and model outputs to a comparable feature space."""

    @abstractmethod
    def transform(self, x: ArrayLike, model_outputs: ArrayLike) -> np.ndarray:
        """Project samples into a 2D consensus representation."""
        raise NotImplementedError


class SpaceProjectorFactory(BaseFactory):
    """Registry-backed factory for projector implementations."""
