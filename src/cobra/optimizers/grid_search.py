from __future__ import annotations
from typing import Callable, Optional, Dict, Any, Sequence

import numpy as np
from tqdm import tqdm

from cobra.optimizers.base import BaseOptimizer, OptimizerFactory


@OptimizerFactory.register("grid", "grid_search", "grid search")
class GridSearchOptimizer(BaseOptimizer):
    """
    Grid Search Optimizer for bandwidth selection.

    Purpose
    -------
    Minimizes:
        h* = argmin_{h in grid} L(h)

    This is the baseline optimizer used in COBRA / MixCOBRA papers.

    Characteristics
    ---------------
    - Exhaustive search over predefined bandwidth list
    - No gradient assumptions
    - Stable reference baseline for comparisons
    """

    def __init__(
        self,
        grid: Optional[Sequence[float]] = None,
        show_progress: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.grid = np.array(grid) if grid is not None else np.linspace(0.0001, 3.0, 100)
        self.show_progress = show_progress

    # --------------------------------------------------
    # main optimization
    # --------------------------------------------------
    def step(
        self,
        loss_fn: Callable[[float], float],
        init_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        loss_fn : Callable
            Black-box loss L(h)

        Returns
        -------
        dict
            {
                'opt_bandwidth': best h,
                'opt_index': index in grid,
                'grid_errors': list of losses
            }
        """

        n_iter = len(self.grid)
        errors = np.zeros(n_iter, dtype=np.float32)

        # --------------------------------------------------
        # evaluate grid
        # --------------------------------------------------
        iterator = tqdm(range(n_iter), desc="Grid Search") if self.show_progress else range(n_iter)

        for i in iterator:
            h = float(self.grid[i])
            errors[i] = loss_fn(h)

        # --------------------------------------------------
        # best selection
        # --------------------------------------------------
        best_idx = int(np.argmin(errors))
        best_h = float(self.grid[best_idx])
        best_loss = float(errors[best_idx])

        return {
            "opt_method": "grid",
            "opt_bandwidth": best_h,
            "opt_index": best_idx,
            "opt_risk": best_loss,
            "grid_errors": errors,
        }
