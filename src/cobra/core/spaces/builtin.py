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


@SpaceProjectorFactory.register("discrete", "prediction_space")
class DiscreteProjector(BaseSpaceProjector):
    """Use only model outputs as the consensus-space coordinates."""

    def transform(self, x: ArrayLike, model_outputs: ArrayLike) -> np.ndarray:
        _ = x
        return _to_2d(model_outputs)


@SpaceProjectorFactory.register("tradeoff", "mixcobra")
class TradeOffProjector(BaseSpaceProjector):
    """Concatenate input features and outputs with an adjustable trade-off."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)

    def transform(self, x: ArrayLike, model_outputs: ArrayLike) -> np.ndarray:
        x2d = _to_2d(x)
        y2d = _to_2d(model_outputs)
        if x2d.shape[0] != y2d.shape[0]:
            raise ValueError("x and model_outputs must have the same number of rows.")
        return np.hstack([x2d * self.alpha, y2d * self.beta])

@SpaceProjectorFactory.register("prediction")
class PredictionProjector(BaseSpaceProjector):
    def transform(self, x: ArrayLike, model_outputs: ArrayLike) -> np.ndarray:
        _ = x
        return _to_2d(model_outputs)
