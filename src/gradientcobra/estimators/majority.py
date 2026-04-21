import numpy as np

from gradientcobra.estimators.base import BaseEstimator
from gradientcobra.factories.estimator import EstimatorFactory

@EstimatorFactory.register("majority", "dummy_classifier")
class MajorityEstimator(BaseEstimator):
    """
    Simple baseline estimator that always predicts the majority class.

    Used for:
    - sanity checking COBRA pipeline
    - baseline comparisons
    """

    def fit(self, X, y: np.ndarray, **kwargs):
        values, counts = np.unique(y, return_counts=True)
        self.majority_class_ = values[np.argmax(counts)]
        return self

    def predict(self, X, **kwargs) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=self.majority_class_)
