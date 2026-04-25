"""
Data preprocessing utilities.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from sklearn.utils import shuffle as sklearn_shuffle


def data_split_overlap(
    X: np.ndarray,
    y: np.ndarray,
    split: float = 0.5,
    overlap: float = 0.0,
    shuffle: bool = True,
    random_state: int = None
)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split (X, y) into D_k and D_l.

    Returns
    -------
    X_k, y_k, X_l, y_l, iloc_k, iloc_l
    """

    if not (0 < split < 1):
        raise ValueError(f"`split` must be in (0,1), got {split}")

    if not (0 <= overlap < 1):
        raise ValueError(f"`overlap` must be in [0,1), got {overlap}")

    if overlap >= split:
        raise ValueError("`overlap` must be smaller than `split`")
    
    n       = len(y)
    indices = np.arange(n)

    if shuffle:
        indices = sklearn_shuffle(indices, random_state=random_state)

    k1 = int(n * (split - overlap / 2))
    k2 = int(n * (split + overlap / 2))

    k1 = max(0, min(k1, n))
    k2 = max(0, min(k2, n))

    iloc_k = indices[:k2].astype(np.int64)
    iloc_l = indices[k1:].astype(np.int64)

    X_k, y_k = X[iloc_k], y[iloc_k]
    X_l, y_l = X[iloc_l], y[iloc_l]

    return X_k, y_k, X_l, y_l, iloc_k, iloc_l

def compute_normalization_constant(
    data: np.ndarray,
    norm_constant: Optional[float] = None,
    scale_factor: float = 1.0,
    M: int | None = None
) -> np.ndarray:
    """
    Compute a normalization/scaling constant for an array.
    """
    M = M or 1.0
    max_val = np.max(np.abs(data)) + 1e-12
    c = norm_constant if norm_constant is not None else scale_factor
    return c / (max_val * M)