"""Concrete splitters aligned with common COBRA workflows."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import KFold, train_test_split

from cobra.utils.preprocessing import data_split_overlap

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
@SplitterFactory.register("split_overlap")
class OverlapSplitter(BaseDataSplitter):
    """
    Split dataset into two overlapping partitions.

    This splitter creates:
    - A "train" index set
    - An "aggregation/calibration" index set

    The overlap controls how many samples are shared between the two sets.
    """

    def __init__(
        self,
        split: float = 0.5,
        overlap: float = 0.0,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        self.split = float(split)
        self.overlap = float(overlap)
        self.shuffle = shuffle
        self.random_state = random_state

    def _shuffle_indices(self, indices: np.ndarray) -> np.ndarray:
        """Internal deterministic shuffle."""
        if not self.shuffle:
            return indices

        rng = np.random.default_rng(self.random_state)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        return shuffled

    def split(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform overlapping split without external helpers.
        """
        n = np.asarray(x).shape[0]
        indices = np.arange(n)

        indices = self._shuffle_indices(indices)

        # ---- boundary validation ----
        if not (0 < self.split < 1):
            raise ValueError(f"`split` must be in (0,1), got {self.split}")

        if not (0 <= self.overlap < 1):
            raise ValueError(f"`overlap` must be in [0,1), got {self.overlap}")

        if self.overlap >= self.split:
            raise ValueError("`overlap` must be smaller than `split`")

        # ---- compute cut points ----
        k1 = int(n * (self.split - self.overlap / 2))
        k2 = int(n * (self.split + self.overlap / 2))

        # clamp to valid range
        k1 = max(0, min(k1, n))
        k2 = max(0, min(k2, n))

        # ---- construct overlapping sets ----
        idx_train = indices[:k2].astype(np.int64)
        idx_agg = indices[k1:].astype(np.int64)

        return idx_train, idx_agg
