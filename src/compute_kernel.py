"""Row-wise mean of stacked model predictions."""

from __future__ import annotations

import numpy as np


def ensemble_mean(predictions: np.ndarray, n_models: int, n_samples: int) -> np.ndarray:
    p = np.asarray(predictions, dtype=float).reshape(n_models, n_samples)
    return p.mean(axis=0)
