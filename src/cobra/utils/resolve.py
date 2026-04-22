from __future__ import annotations

from typing import Any, Iterable

from cobra.core.aggregators.base import AggregatorFactory
from cobra.core.distances.base import DistanceFactory
from cobra.core.estimators.base import EstimatorFactory
from cobra.core.kernels.base import KernelFactory
from cobra.core.losses.base import LossFactory
from cobra.core.optimizers.base import OptimizerFactory
from cobra.core.spaces.base import SpaceProjectorFactory
from cobra.core.splitters.base import SplitterFactory


def resolve_from_estimators(
    estimators: list[Any] | str | None,
    estimators_params: dict[str, Any] | None,
    default_estimators: list[str],
) -> list[Any]:
    """
    Resolve estimator definitions into instantiated estimators.

    This function supports:
    - factory string aliases (e.g. "linear", "ridge")
    - sklearn-style wrapped estimators via EstimatorFactory
    - default fallback pool
    - single string or list input

    Parameters
    ----------
    estimators:
        User-defined estimator list or single alias.
        If None -> default_estimators is used.

    estimators_params:
        Global parameter dictionary passed to estimator constructors.

    default_estimators:
        List of default estimator aliases.

    Returns
    -------
    list
        List of instantiated estimator objects.
    """

    if estimators is None:
        estimators = default_estimators

    if isinstance(estimators, str):
        estimators = [estimators]

    if not isinstance(estimators, Iterable):
        raise TypeError("estimators must be a list, string, or None")

    resolved: list[Any] = []

    for est in estimators:
        if isinstance(est, str):
            name = est.lower()

            try:
                resolved_est = EstimatorFactory.create(
                    name,
                    **(estimators_params or {}),
                )
            except Exception as e:
                raise ValueError(
                    f"Unknown estimator '{est}'. "
                    f"Available: {EstimatorFactory.available()}"
                ) from e

        else:
            resolved_est = est

        resolved.append(resolved_est)

    return resolved

def resolve_from_kernel(
    kernel: str | Any,
    kernel_params: dict[str, Any] | None,
):
    return KernelFactory.create(kernel, **(kernel_params or {}))

def resolve_from_splitter(
    splitter: str | Any,
    splitter_params: dict[str, Any] | None,
):
    return SplitterFactory.create(splitter, **(splitter_params or {}))

def resolve_from_optimizer(
    optimizer: str | Any,
    optimizer_params: dict[str, Any] | None,
):
    return OptimizerFactory.create(optimizer, **(optimizer_params or {}))

def resolve_from_loss(
    loss: str | Any,
    loss_params: dict[str, Any] | None,
):
    return LossFactory.create(loss, **(loss_params or {}))

def resolve_from_space(
    space: str | Any,
    space_params: dict[str, Any] | None,
):
    return SpaceProjectorFactory.create(space, **(space_params or {}))

def resolve_from_distance(
    distance: str | Any,
    distance_params: dict[str, Any] | None,
):
    return DistanceFactory.create(distance, **(distance_params or {}))

def resolve_from_aggregator(
    aggregator: str | Any,
    aggregator_params: dict[str, Any] | None,
):
    return AggregatorFactory.create(aggregator, **(aggregator_params or {}))

def resolve_from_optimizer(
    optimizer: str | Any,
    optimizer_params: dict[str, Any] | None,
):
    return OptimizerFactory.create(optimizer, **(optimizer_params or {}))