import numpy as np
from itertools import product

from cobra.core.optimizers.search.base import BaseSearchOptimizer


class GridSearchOptimizer(BaseSearchOptimizer):
    """
    Exhaustive grid search optimizer.
    """

    def __init__(self, param_grid: dict, verbose=False):
        self.param_grid = param_grid
        self.verbose = verbose

    def search_space(self):
        return self.param_grid

    def sample(self):
        raise NotImplementedError("GridSearch does not use sampling.")

    def __call__(self, objective, *args, **kwargs):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        best_params = None
        best_score = float("inf")
        history = []

        for combo in product(*values):
            params = dict(zip(keys, combo))
            score = objective(list(params.values()))

            history.append((params, score))

            if score < best_score:
                best_score = score
                best_params = params

        return np.array(list(best_params.values())), history
