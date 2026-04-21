
from typing import Dict

from cobra.kernels.base import BaseKernel
from cobra.aggregators.base import BaseAggregator
from cobra.distances.base import BaseDistance
from cobra.estimators.base import BaseEstimator

from cobra.factories.aggregator import AggregatorFactory
from cobra.factories.distance import DistanceFactory
from cobra.factories.estimator import EstimatorFactory
from cobra.factories.kernel import KernelFactory


def resolve_from_estimators(estimators, estimators_params):
    result : Dict[str, BaseEstimator] = {}

    for estimator in estimators:
        if isinstance(estimator, BaseEstimator):
            name = type(estimator).__name__.lower()
            result[name] = estimator
        else:
            params = estimators_params.get(estimator)
            result[estimator] = EstimatorFactory.create(estimator, **(params or {}))

    return result

def resolve_from_distance(distance, distance_params):
    if isinstance(distance, BaseDistance):
        return distance
    if isinstance(distance, str):
        return DistanceFactory.create(distance, **(distance_params or {}))

def resolve_from_kernel(kernel, kernel_params):
    if isinstance(kernel, BaseKernel):
        return kernel
    if isinstance(kernel, str):
        return KernelFactory.create(kernel, **(kernel_params or {}))

def resolve_from_aggregator(aggregator, aggregator_params):
    if isinstance(aggregator, BaseAggregator):
        return aggregator
    if isinstance(aggregator, str):
        return AggregatorFactory.create(aggregator, **(aggregator_params or {}))
