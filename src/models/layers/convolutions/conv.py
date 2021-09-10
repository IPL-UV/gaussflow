"""Original Code:
    https://github.com/ehoogeboom/convolution_exponential_and_sylvester/blob/main/models/transformations/conv1x1.py
"""
from typing import Iterable, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import FrEIA.modules as Fm
from src.models.layers.utils import construct_householder_matrix


class Conv1x1(Fm.InvertibleModule):
    def __init__(self, dims_in: Iterable[Tuple[int]]):
        super().__init__(dims_in)

        # extract dimensions
        self.n_channels = dims_in[0][0]
        self.H = dims_in[0][1]
        self.W = dims_in[0][2]

        w_np = np.random.randn(self.n_channels, self.n_channels)
        q_np = np.linalg.qr(w_np)[0]

        self.V = torch.nn.Parameter(torch.from_numpy(q_np.astype("float32")))

    def forward(self, x, rev=False, jac=True):
        x = x[0]
        n_samples, *_ = x.size()

        log_jac_det = torch.zeros(n_samples, dtype=x.dtype, device=x.device)

        w = self.V
        d_ldj = self.H * self.W * torch.slogdet(w)[1]

        if not rev:
            w = w.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1)

            log_jac_det += d_ldj

        else:
            w_inv = torch.inverse(w)
            w_inv = w_inv.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(x, w_inv, bias=None, stride=1, padding=0, dilation=1, groups=1)

            log_jac_det -= d_ldj

        return (z,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims
