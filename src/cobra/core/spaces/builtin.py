"""Concrete projectors capturing COBRA and MIXCOBRA geometry."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .base import BaseSpaceProjector, SpaceProjectorFactory


def _to_2d(arr: ArrayLike) -> np.ndarray:
    """Normalize array-like input to 2D."""
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    if out.ndim != 2:
        raise ValueError("Expected 1D or 2D input.")
    return out

@SpaceProjectorFactory.register("combine_classifier")
class CombineClassifierSpaceProjector(BaseSpaceProjector):
    """
    COBRA space:
    purely prediction space geometry.
    """

    def transform(self, x: ArrayLike, model_outputs: ArrayLike) -> np.ndarray:
        return _to_2d(model_outputs)

@SpaceProjectorFactory.register("gradientcobra", "prediction_only")
class PredictionOnlyProjector(BaseSpaceProjector):
    """
    GradientCOBRA space:
    purely prediction space geometry.
    """

    def transform(self, x: ArrayLike, model_outputs: ArrayLike) -> np.ndarray:
        return _to_2d(model_outputs)

@SpaceProjectorFactory.register("mixcobra", "tradeoff")
class MixCOBRASpaceProjector(BaseSpaceProjector):
    """
    MixCOBRA joint space:
    combines input + prediction geometry.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, one_parameter: bool = False) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.one_parameter = one_parameter

    def transform(self, x: ArrayLike, model_outputs: ArrayLike) -> np.ndarray:
        x = _to_2d(x)
        y = _to_2d(model_outputs)

        if x.shape[0] != y.shape[0]:
            raise ValueError("x and model_outputs must match rows")
        
        if self.one_parameter:
            if self.beta != 0.0:
                raise ValueError("beta must be zero when one_parameter is True")
            return self.alpha * np.hstack([x, y])

        return np.hstack([
            self.alpha * x,
            self.beta * y
        ])