import torch
from torch import nn
import FrEIA.modules as Fm
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.distributions as dist
from nflows import transforms
from nflows.utils import sum_except_batch
from einops import repeat
import numpy as np
from sklearn.mixture import GaussianMixture


class NFlowsLayer(Fm.InvertibleModule):
    def __init__(
        self,
        dims_in,
        transform,
        name="linear",
    ):
        super().__init__(dims_in)

        self.n_features = dims_in[0][0]
        self.transform = transform
        self.name = name

    def forward(self, x, rev=False, jac=True):
        x = x[0]
        if rev:

            z, log_det = self.linear_transform.inverse(x)
            # print(f"Mix (Out): {z.min(), z.max()}")

        else:
            # print(x.shape)
            z, log_det = self.linear_transform.forward(x)

        return (z,), log_det

    def output_dims(self, input_dims):
        return input_dims


def construct_householder_matrix(V):
    n_reflections, n_channels = V.shape

    I = torch.eye(n_channels, dtype=V.dtype, device=V.device)

    Q = I

    for i in range(n_reflections):
        v = V[i].view(n_channels, 1)

        vvT = torch.matmul(v, v.t())
        vTv = torch.matmul(v.t(), v)
        Q = torch.matmul(Q, I - 2 * vvT / vTv)

    return Q
