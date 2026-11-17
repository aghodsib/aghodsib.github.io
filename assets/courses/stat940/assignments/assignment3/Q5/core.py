"""Complete the graph operations. Node features are stored in columns."""

import math
import torch
from torch import nn


def normalized_adjacency(A):
    """Return S as a floating torch.Tensor with shape (n,n).

    A is a floating torch.Tensor with shape (n,n), symmetric, with zero diagonal.
    Add one self-loop per node before computing the stated normalization.
    Preserve A's dtype and device.
    """
    raise NotImplementedError("Implement symmetric adjacency normalization.")


class GraphLayer(nn.Module):
    """Graph layer from d input features to f output features.

    Parameters and their fixed initialization are supplied. Complete forward.
    Use activation=None for the identity activation required by the checks.
    """

    def __init__(self, d, f, activation=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty((f, d)))
        self.b = nn.Parameter(torch.empty(f))
        self.activation = activation
        # This is the same default affine initialization used by nn.Linear.
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.uniform_(self.b, -1 / math.sqrt(d), 1 / math.sqrt(d))

    def forward(self, H, S):
        """Return shape (f,n) from H of shape (d,n) and S of shape (n,n).

        Add the learnable bias after aggregation, then apply self.activation
        if it is not None. Preserve differentiability through self.W/self.b.
        """
        raise NotImplementedError("Implement the prescribed graph layer.")
