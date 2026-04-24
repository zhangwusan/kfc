
from __future__ import annotations
import numpy as np

from cobra.core.optimizers.base import tqdm, trange
from cobra.core.optimizers.gradient.base import BaseGradientOptimizer, GradientOptimizerFactory


@GradientOptimizerFactory.register("grad", "gradient_descent")
class GradientDescentOptimizer(BaseGradientOptimizer):
    def __init__(
        self,
        learning_rate=0.01,
        max_iter=100,
        tol=1e-6,
        eps=1e-8,
        verbose=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.verbose = verbose
        

    def gradient(self, objective, params):
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_eps = np.array(params, copy=True)
            params_eps[i] += self.eps
            grad[i] = (objective(params_eps) - objective(params)) / self.eps
        return grad
    
    def step(self, objective, params):
        grad = self.gradient(objective, params)
        new_params = params - self.learning_rate * grad
        return new_params
    
    def __call__(self, objective, params):
        params = np.array(params, dtype=float)
        history = []

        iterator = (
            range(self.max_iter) if not self.verbose else tqdm(range(self.max_iter), desc="GD")
        )

        for i in iterator:
            grad = self.gradient(objective, params)
            params = self.step(objective, params)

            history.append(grad.copy())

            if np.linalg.norm(grad) < self.tol:
                break
            if self.verbose:
                iterator.set_description(f"GD iter: {i} / grad : {np.linalg.norm(grad):.4f}")

        return params, history
