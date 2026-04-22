"""Concrete splitters aligned with common COBRA workflows."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import KFold, train_test_split

from .base import BaseDataSplitter, SplitterFactory


@SplitterFactory.register("holdout", "random_holdout")
class RandomHoldoutSplitter(BaseDataSplitter):
    """Randomly split indices into train and calibration sets."""

    def __init__(self, calibration_size: float = 0.5, random_state: int | None = None) -> None:
        self.calibration_size = float(calibration_size)
        self.random_state = random_state

    def split(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        n_samples = np.asarray(x).shape[0]
        indices = np.arange(n_samples)
        train_idx, cal_idx = train_test_split(
            indices,
            test_size=self.calibration_size,
            random_state=self.random_state,
            shuffle=True,
        )
        return np.asarray(train_idx), np.asarray(cal_idx)


@SplitterFactory.register("kfold")
class KFoldSplitter(BaseDataSplitter):
    """Use one fold as calibration and the rest as train indices."""

    def __init__(self, n_splits: int = 5, fold_index: int = 0, random_state: int | None = None) -> None:
        self.n_splits = int(n_splits)
        self.fold_index = int(fold_index)
        self.random_state = random_state

    def split(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        _ = y
        n_samples = np.asarray(x).shape[0]
        indices = np.arange(n_samples)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        folds = list(kf.split(indices))
        if not 0 <= self.fold_index < len(folds):
            raise ValueError("fold_index is out of range for n_splits.")
        train_idx, cal_idx = folds[self.fold_index]
        return np.asarray(train_idx), np.asarray(cal_idx)
