"""Exact supplied data generator. Observations are COLUMNS, labels shape (n,)."""
import numpy as np


def make_split(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    angle = rng.uniform(0.0, np.pi, size=n)
    x0 = np.where(y == 0, np.cos(angle), 1.0 - np.cos(angle))
    x1 = np.where(y == 0, np.sin(angle), 0.5 - np.sin(angle))
    X = np.vstack([x0, x1]) + rng.normal(0.0, 0.18, size=(2, n))
    X = np.vstack([X, 3.0 * rng.normal(size=n)])
    y = np.logical_xor(y, rng.random(n) < 0.05).astype(np.float32)
    return X.astype(np.float32), y


def raw_splits():
    """Return unprocessed train, validation, test pairs in that order."""
    return (make_split(256, 9401), make_split(128, 9402),
            make_split(512, 9403))
