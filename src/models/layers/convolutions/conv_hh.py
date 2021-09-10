"""Original Code:
    https://github.com/ehoogeboom/convolution_exponential_and_sylvester/blob/main/models/transformations/conv1x1.py
"""
from typing import Iterable, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import FrEIA.modules as Fm
from src.models.layers.utils import construct_householder_matrix


class Conv1x1Householder(Fm.InvertibleModule):
    def __init__(
        self,
        dims_in: Iterable[Tuple[int]],
        n_reflections: int = 10,
    ):
        super().__init__(dims_in)

        self.n_reflections = n_reflections

        # extract dimensions
        self.n_channels = dims_in[0][0]
        self.H = dims_in[0][1]
        self.W = dims_in[0][2]

        # initialize matrix
        v_np = np.random.randn(self.n_reflections, self.n_channels)

        self.V = torch.nn.Parameter(torch.from_numpy(v_np.astype("float32")))

    def forward(self, x, rev=False, jac=True):
        x = x[0]

        n_samples, *_ = x.size()

        ldj = torch.zeros(n_samples, dtype=x.dtype, device=x.device)

        Q = construct_householder_matrix(self.V)

        #

        if not rev:
            Q = Q.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(x, Q, bias=None, stride=1, padding=0, dilation=1, groups=1)

        else:
            Q_inv = Q.t()
            Q_inv = Q_inv.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(x, Q_inv, bias=None, stride=1, padding=0, dilation=1, groups=1)

        return (z,), ldj

    def inverse(self, x, rev=False, jac=True):
        return self(x, rev=True, jac=jac)

    def output_dims(self, input_dims):
        return input_dims
