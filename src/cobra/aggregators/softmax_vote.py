from __future__ import annotations
import numpy as np
from gradientcobra.aggregators.base import BaseAggregator
from gradientcobra.factories.aggregator import AggregatorFactory

@AggregatorFactory.register("softmax_vote")
class SoftmaxVoteAggregator(BaseAggregator):
    """
    Soft classification via temperature scaling.
    """

    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def aggregate(self, weights: np.ndarray, y: np.ndarray):
        # convert weights → probabilities
        probs = np.exp(weights / self.temperature)
        probs = probs / (np.sum(probs) + 1e-8)

        unique_classes = np.unique(y)
        class_score = {}

        for c in unique_classes:
            class_score[c] = np.sum(probs[y == c])

        return max(class_score, key=class_score.get)
