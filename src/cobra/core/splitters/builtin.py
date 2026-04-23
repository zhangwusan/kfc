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
    """
    Returns full KFold splits:
    List of (train_idx, val_idx)
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        self.n_splits = int(n_splits)
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, x: ArrayLike, y: ArrayLike):
        n_samples = np.asarray(x).shape[0]
        indices = np.arange(n_samples)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        return [
            (
                np.asarray(train_idx, dtype=np.int64),
                np.asarray(val_idx, dtype=np.int64)
            )
            for train_idx, val_idx in kf.split(indices)
        ]

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
        split_ratio: float = 0.5,
        overlap: float = 0.0,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        self.split_ratio = float(split_ratio)
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
        if not (0 < self.split_ratio < 1):
            raise ValueError(f"`split_ratio` must be in (0,1), got {self.split_ratio}")

        if not (0 <= self.overlap < 1):
            raise ValueError(f"`overlap` must be in [0,1), got {self.overlap}")

        if self.overlap >= self.split_ratio:
            raise ValueError("`overlap` must be smaller than `split_ratio`")

        # ---- compute cut points ----
        k1 = int(n * (self.split_ratio - self.overlap / 2))
        k2 = int(n * (self.split_ratio + self.overlap / 2))

        # clamp to valid range
        k1 = max(0, min(k1, n))
        k2 = max(0, min(k2, n))

        # ---- construct overlapping sets ----
        idx_train = indices[:k2].astype(np.int64)
        idx_agg = indices[k1:].astype(np.int64)

        return idx_train, idx_agg
