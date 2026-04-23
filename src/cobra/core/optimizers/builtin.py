"""
Builtin optimizers for hyperparameter search.

Supports both:
- 1D optimization (e.g., bandwidth or alpha)
- Multi-dimensional optimization (e.g., alpha, beta)

All optimizers accept:
    objective: Callable[[np.ndarray], float]
    initial_value: float | np.ndarray

Return:
    float (1D) or np.ndarray (multi-D)
"""

from __future__ import annotations

import numpy as np
from collections.abc import Callable

from cobra.core.optimizers.base import BaseOptimizer, OptimizerFactory, tqdm


# =========================================================
# Grid Search Optimizer
# =========================================================

@OptimizerFactory.register("grid", "grid_search")
class GridSearchOptimizer(BaseOptimizer):
    """
    Exhaustive grid search optimizer.

    Parameters
    ----------
    grid : array-like or list of tuples, optional
        - If 1D: array of candidate values
        - If multi-D: list of tuples (e.g., [(α, β), ...])
        If None, a default grid is generated.

    show_progress : bool, default=True
        Whether to display tqdm progress bar.
    """

    def __init__(self, grid=None, show_progress: bool = True):
        self.grid = grid
        self.show_progress = show_progress

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        initial_value
    ):
        """
        Run grid search.

        Parameters
        ----------
        objective : callable
            Function mapping parameters → scalar loss.

        initial_value : float or np.ndarray
            Used to infer dimensionality if grid is None.

        Returns
        -------
        best_param : float or np.ndarray
        """

        # --- infer dimension ---
        dim = len(np.atleast_1d(initial_value))

        # --- build default grid ---
        if self.grid is None:
            if dim == 1:
                self.grid = np.linspace(0.01, 10.0, 100)
            else:
                a = np.linspace(0.01, 10.0, 5)
                b = np.linspace(0.01, 10.0, 5)
                self.grid = [(i, j) for i in a for j in b]

        best_x = None
        best_score = float("inf")

        iterator = tqdm(self.grid, desc="Grid Search", disable=not self.show_progress)

        for val in iterator:
            x = np.atleast_1d(val).astype(float)
            score = objective(x)

            if score < best_score:
                best_score = score
                best_x = x

        if best_x is None:
            raise RuntimeError("Grid search failed to find a valid parameter.")

        return best_x if len(best_x) > 1 else best_x[0]


# =========================================================
# Gradient Descent Optimizer
# =========================================================

@OptimizerFactory.register("grad", "gradient", "gradient_descent")
class GradientDescentOptimizer(BaseOptimizer):
    """
    Finite-difference gradient descent optimizer.

    Supports both scalar and vector parameter optimization.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Initial learning rate.

    max_iter : int, default=100
        Maximum number of iterations.

    tol : float, default=1e-6
        Gradient norm stopping criterion.

    eps : float, default=1e-5
        Finite difference step size.

    speed : str, default="constant"
        Learning rate schedule:
            - "constant"
            - "inverse"
            - "exponential"
            - "log"

    show_progress : bool, default=True
        Whether to show tqdm progress bar.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
        eps: float = 1e-5,
        speed: str = "constant",
        show_progress: bool = True
    ):
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.eps = float(eps)
        self.speed = speed
        self.show_progress = bool(show_progress)

        self.schedules = {
            "constant": lambda t: self.learning_rate,
            "inverse": lambda t: self.learning_rate / (1 + t),
            "exponential": lambda t: self.learning_rate * (0.9 ** t),
            "log": lambda t: self.learning_rate * np.log(2 + t),
        }

        if self.speed not in self.schedules:
            raise ValueError(f"Unknown speed schedule: {self.speed}")

    # -----------------------------------------------------

    def _gradient(self, objective, x: np.ndarray) -> np.ndarray:
        """
        Compute finite-difference gradient.
        """
        x = np.atleast_1d(x).astype(float)
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x1 = x.copy()
            x2 = x.copy()
            x1[i] += self.eps
            x2[i] -= self.eps

            grad[i] = (objective(x1) - objective(x2)) / (2 * self.eps)

        return grad

    # -----------------------------------------------------

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        initial_value
    ):
        """
        Run gradient descent optimization.

        Parameters
        ----------
        objective : callable
            Function mapping parameters → scalar loss.

        initial_value : float or np.ndarray
            Initial guess.

        Returns
        -------
        best_param : float or np.ndarray
        """

        x = np.atleast_1d(initial_value).astype(float)
        best_x = x.copy()
        best_score = objective(x)

        schedule = self.schedules[self.speed]

        iterator = tqdm(
            range(self.max_iter),
            desc="Gradient Descent",
            disable=not self.show_progress
        )

        for t in iterator:
            grad = self._gradient(objective, x)

            # --- stopping criterion ---
            if np.linalg.norm(grad) < self.tol:
                break

            step = schedule(t)
            x_new = x - step * grad

            # --- enforce positivity (important for α, β, bandwidth) ---
            x_new = np.maximum(x_new, 1e-8)

            score = objective(x_new)

            if score < best_score:
                best_score = score
                best_x = x_new.copy()

            x = x_new

            if self.show_progress:
                iterator.set_postfix(
                    score=f"{score:.4f}",
                    best=f"{best_score:.4f}",
                    grad=f"{np.linalg.norm(grad):.4f}"
                )

        return best_x if len(best_x) > 1 else best_x[0]
