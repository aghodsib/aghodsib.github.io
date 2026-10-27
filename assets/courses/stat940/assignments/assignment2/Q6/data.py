"""Fixed, label-free training data protocol for STAT 940 Assignment 2 Q6."""
import numpy as np
import torch

CENTERS = np.array([[-2., -2.], [-2., 2.], [2., -2.], [2., 2.]], dtype=np.float64)

def make_data(n, seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, size=n)
    X = CENTERS[labels] + 0.20 * rng.standard_normal((n, 2))
    return torch.from_numpy(X.astype(np.float32)), labels

