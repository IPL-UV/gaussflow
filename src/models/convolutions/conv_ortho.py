"""Original Code:
    https://github.com/ehoogeboom/convolution_exponential_and_sylvester/blob/main/models/transformations/conv1x1.py
"""
import einops.layers.torch as eintorch
import numpy as np
import torch
import torch.nn.functional as F
from nflows.transforms import Transform


class Conv1x1Householder(Transform):
    def __init__(self, n_channels: int, H: int, W: int, n_reflections: int = 10):
        super().__init__()
        self.n_channels = n_channels
        self.n_reflections = n_reflections
        self.H = H
        self.W = W
        self.flatten_layer = eintorch.Rearrange("b c h w -> b (c h w)")
        self.image_layer = eintorch.Rearrange(
            "b (c h w) -> b c h w", c=n_channels, h=H, w=W
        )
        v_np = np.random.randn(n_reflections, n_channels)

        self.V = torch.nn.Parameter(torch.from_numpy(v_np.astype("float32")))

    def contruct_Q(self):
        I = torch.eye(self.n_channels, dtype=self.V.dtype, device=self.V.device)
        Q = I

        for i in range(self.n_reflections):
            v = self.V[i].view(self.n_channels, 1)

            vvT = torch.matmul(v, v.t())
            vTv = torch.matmul(v.t(), v)
            Q = torch.matmul(Q, I - 2 * vvT / vTv)

        return Q

    def forward(self, x, context=None, reverse=False):
        n_samples, n_features = x.size()

        ldj = torch.zeros(n_samples, dtype=x.dtype)

        # create image
        x = self.image_layer(x)

        Q = self.contruct_Q()

        if not reverse:
            Q = Q.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(x, Q, bias=None, stride=1, padding=0, dilation=1, groups=1)

        else:
            Q_inv = Q.t()
            Q_inv = Q_inv.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(x, Q_inv, bias=None, stride=1, padding=0, dilation=1, groups=1)

        z = self.flatten_layer(z)
        return z, ldj

    def inverse(self, z, context=None):
        return self(z, context=context, reverse=True)
