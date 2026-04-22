"""Concrete optimizers for hard and differentiable search spaces."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .base import BaseOptimizer, OptimizerFactory, tqdm, trange


@OptimizerFactory.register("grid", "grid_search")
class GridSearchOptimizer(BaseOptimizer):
    """Evaluate a fixed grid and return the best scalar parameter."""

    def __init__(self, low: float = 1e-3, high: float = 5.0, num: int = 50) -> None:
        self.low = float(low)
        self.high = float(high)
        self.num = int(num)

    def optimize(self, objective: Callable[[float], float], initial_value: float) -> float:
        _ = initial_value
        candidates = np.linspace(self.low, self.high, self.num)
        scores = [objective(float(v)) for v in candidates]
        return float(candidates[int(np.argmin(scores))])

@OptimizerFactory.register("gradient", "gradient_descent")
class GradientDescentOptimizer(BaseOptimizer):
    """Finite-difference gradient descent with tqdm progress display."""

    def __init__(
        self,
        lr: float = 0.05,
        max_iter: int = 200,
        eps: float = 1e-5,
        verbose: bool = True
    ) -> None:
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.eps = float(eps)
        self.verbose = bool(verbose)

    def optimize(self, objective: Callable[[float], float], initial_value: float) -> float:
        x = float(initial_value)

        pbar = trange(
            self.max_iter,
            desc="Optimizing Gradient",
            disable=not self.verbose
        )

        best_x = x
        best_score = objective(x)

        for t in pbar:

            grad = (
                objective(x + self.eps) - objective(x - self.eps)
            ) / (2.0 * self.eps)

            x = x - self.lr * grad
            score = objective(x)

            if score < best_score:
                best_score = score
                best_x = x

            pbar.set_postfix(
                x=f"{x:.4f}",
                score=f"{score:.4f}",
                best=f"{best_score:.4f}"
            )

        return best_x
