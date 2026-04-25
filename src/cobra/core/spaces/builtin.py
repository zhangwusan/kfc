"""
Normalized spaces
"""

from __future__ import annotations

from cobra.core.spaces.base import BaseSpaceNormalizer, SpaceNormalizerFactory
from cobra.utils.preprocessing import compute_normalization_constant

@SpaceNormalizerFactory.register("identity")
class IdentitySpaceNormalizer(BaseSpaceNormalizer):
    def transform(self, X, model_outputs):
        return X, model_outputs

@SpaceNormalizerFactory.register("gradientcobra")
class GradientCOBRASpaceNormalizer(BaseSpaceNormalizer):
    def __init__(self, norm_constant = None):
        self.norm_constant = norm_constant

    def transform(self, X, model_outputs):
        M = model_outputs.shape[1]
        normalize_constant = compute_normalization_constant(
            model_outputs,
            self.norm_constant,
            scale_factor=30.0,
            M=M
        )

        Y = model_outputs / normalize_constant

        return X, Y
    
@SpaceNormalizerFactory.register("mixcobra")
class MixCOBRASpaceNormalizer(BaseSpaceNormalizer):
    def __init__(self, norm_constant_x = None, norm_constant_y = None):
        self.norm_constant_x = norm_constant_x
        self.norm_constant_y = norm_constant_y

    def transform(self, X, model_outputs):
        M = model_outputs.shape[1]
        normalize_constant_x = compute_normalization_constant(
            X,
            self.norm_constant_x,
            scale_factor=30.0,
            M=M
        )
        normalize_constant_y = compute_normalization_constant(
            model_outputs,
            self.norm_constant_y,
            scale_factor=30.0,
            M=M
        )
        X = X / normalize_constant_x
        Y = model_outputs / normalize_constant_y

        return X, Y
