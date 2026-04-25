"""Base interface for projecting data into consensus spaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseSpaceNormalizer(ABC):
    @abstractmethod
    def transform(
        self,
        X: np.ndarray,
        model_outputs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return normalized feature space and prediction space.
        Returns
        -------
        X_norm : np.ndarray
        Y_norm : np.ndarray
        """
        pass

class SpaceNormalizerFactory(BaseFactory):
    """Registry-backed factory for normalizer implementations."""
