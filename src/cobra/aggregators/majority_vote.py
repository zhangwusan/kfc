
from collections import Counter
import numpy as np

from cobra.aggregators.base import BaseAggregator
from cobra.factories.aggregator import AggregatorFactory



@AggregatorFactory.register("majority_vote", "cobra_vote")
class MajorityVoteAggregator(BaseAggregator):
    """
    Hard voting over selected consensus set.

    Used in original COBRA.
    """

    def aggregate(self, weights: np.ndarray, y: np.ndarray):
        # weights ignored in COBRA (binary selection only)
        if len(y) == 0:
            return None

        counts = Counter(y)
        max_count = max(counts.values())
        candidates = [k for k, v in counts.items() if v == max_count]

        return min(candidates)  # tie-breaking rule
