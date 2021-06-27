"""Original Code:
    https://github.com/ehoogeboom/convolution_exponential_and_sylvester/blob/main/models/transformations/conv1x1.py
"""
import einops.layers.torch as eintorch
import numpy as np
import torch
import torch.nn.functional as F
from nflows.transforms import Transform
import FrEIA.modules as Fm


class Conv1x1(Fm.InvertibleModule):
    def __init__(self, dims_in: List[int], n_channels: int, H: int, W: int):
        super().__init__(dims_in)
        self.n_channels = dims_in[0]

        self.H = dims_in[1]
        self.W = dims_in[2]


        w_np = np.random.randn(self.n_channels, self.n_channels)
        q_np = np.linalg.qr(w_np)[0]


        self.V = torch.nn.Parameter(torch.from_numpy(q_np.astype("float32")))

    def forward(self, x, rev=False, jac=True):
        x = x[0]
        n_samples, n_features = x.size()



        log_jac_det = torch.zeros(n_samples, dtype=x.dtype)

        w = self.V
        d_ldj = self.H * self.W * torch.slogdet(w)[1]

        if not reverse:
            w = w.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1)

            log_jac_det += d_ldj

            return (x,), log_jac_det
        else:
            w_inv = torch.inverse(w)
            w_inv = w_inv.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(x, w_inv, bias=None, stride=1, padding=0, dilation=1, groups=1)

            log_jac_det -= d_ldj


            return (x,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims