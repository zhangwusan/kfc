"""Base interface for creating train/calibration splits."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from cobra.core.factory import BaseFactory


class BaseDataSplitter(ABC):
    """Split input arrays while preserving index alignment."""

    @abstractmethod
    def split(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return two index arrays: train indices and calibration indices."""
        raise NotImplementedError


class SplitterFactory(BaseFactory):
    """Registry-backed factory for splitter implementations."""
