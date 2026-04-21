from __future__ import annotations
from typing import Callable, Optional, Dict, Any

import numpy as np
from tqdm import trange

from cobra.optimizers.base import BaseOptimizer, OptimizerFactory


@OptimizerFactory.register("gd", "grad", "gradient_descent")
class GradientDescentOptimizer(BaseOptimizer):
    """
    COBRA-style Gradient Descent Optimizer (legacy-compatible version).

    This version preserves:
    - adaptive learning rate scaling (r0 trick)
    - sign-change damping
    - bandwidth trace logging
    - kappa-style stopping condition behavior
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iters: int = 300,
        speed: str = "constant",
        epsilon: float = 1e-2,
        n_tries: int = 5,
        precision: float = 1e-7,
        start: Optional[float] = None,
        show_progress: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.speed = speed
        self.epsilon = epsilon
        self.n_tries = n_tries
        self.precision = precision
        self.start = start
        self.show_progress = show_progress

        # learning rate schedules (same as old implementation)
        self.schedules = {
            "constant": lambda t, lr: lr,
            "linear": lambda t, lr: t * lr,
            "log": lambda t, lr: np.log1p(t) * lr,
            "sqrt_root": lambda t, lr: np.sqrt(1 + t) * lr,
            "quad": lambda t, lr: (1 + t**2) * lr,
            "exp": lambda t, lr: np.exp(t) * lr,
        }

        if self.speed not in self.schedules:
            raise ValueError(f"Unknown schedule: {self.speed}")

    # --------------------------------------------------
    # gradient (finite difference)
    # --------------------------------------------------
    def _gradient(self, loss_fn: Callable[[float], float], h: float) -> float:
        eps = self.precision
        return (loss_fn(h + eps) - loss_fn(h - eps)) / (2 * eps)

    # --------------------------------------------------
    # schedule
    # --------------------------------------------------
    def _schedule(self, t: int, lr: float) -> float:
        return self.schedules[self.speed](t, lr)

    # --------------------------------------------------
    # main optimizer (LEGACY-COMPATIBLE)
    # --------------------------------------------------
    def step(
        self,
        loss_fn: Callable[[float], float],
        init_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Returns
        -------
        dict (legacy format compatible with COBRA):
            {
                'opt_bandwidth': float,
                'bandwidth_collection': list,
                'gradients': list,
                'opt_method': 'grad'
            }
        """

        # -----------------------------
        # INIT (same as old)
        # -----------------------------
        if init_value is None:
            candidates = np.linspace(0.0001, 3.0, self.n_tries)
            losses = [loss_fn(h) for h in candidates]
            bw0 = float(candidates[np.argmin(losses)])
        else:
            bw0 = float(init_value)

        grad = self._gradient(loss_fn, bw0)
        grad0 = grad

        # same trick: scale learning rate
        r0 = self.learning_rate / (abs(grad) + 1e-12)
        rate = self._schedule

        collect_bw = []
        gradients = []

        test_threshold = np.inf

        # -----------------------------
        # progress bar
        # -----------------------------
        iterator = trange(self.max_iters, desc="GD Optimization") if self.show_progress else range(self.max_iters)

        # -----------------------------
        # LOOP
        # -----------------------------
        for t in iterator:

            lr_t = rate(t, r0)

            bw = bw0 - lr_t * grad

            # stability (same as old)
            if bw <= 0 or np.isnan(bw):
                bw = bw0 * 0.95

            # sign flip damping (IMPORTANT legacy behavior)
            if t > 3 and np.sign(grad) * np.sign(grad0) < 0:
                r0 *= 0.99

            # update state
            bw0, grad0 = bw, grad

            # recompute gradient
            grad = self._gradient(loss_fn, bw0)

            # stopping criterion (same logic)
            test_threshold = abs(grad)

            collect_bw.append(float(bw0))
            gradients.append(float(grad))

            if self.show_progress:
                iterator.set_description(
                    f"GD | bw={bw0:.4f} | grad={grad:.6f} | stop={test_threshold:.6f}"
                )

            if test_threshold < self.epsilon:
                break

        opt_bw = float(bw0)
        opt_risk = loss_fn(opt_bw)

        return {
            "opt_method": "grad",
            "opt_bandwidth": opt_bw,
            "opt_risk": opt_risk,
            "bandwidth_collection": collect_bw,
            "gradients": gradients,
        }
