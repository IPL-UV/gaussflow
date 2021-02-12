"""Original Code:
    https://github.com/ehoogeboom/convolution_exponential_and_sylvester/blob/main/models/transformations/conv1x1.py
"""
import einops.layers.torch as eintorch
import numpy as np
import torch
import torch.nn.functional as F
from nflows.transforms import Transform


class Conv1x1(Transform):
    def __init__(self, n_channels: int, H: int, W: int):
        super().__init__()
        self.n_channels = n_channels

        self.H = H
        self.W = W
        self.n_channels = n_channels

        self.H = H
        self.W = W

        w_np = np.random.randn(n_channels, n_channels)
        q_np = np.linalg.qr(w_np)[0]

        self.flatten_layer = eintorch.Rearrange("b c h w -> b (c h w)")
        self.image_layer = eintorch.Rearrange(
            "b (c h w) -> b c h w", c=n_channels, h=H, w=W
        )

        self.V = torch.nn.Parameter(torch.from_numpy(q_np.astype("float32")))

    def forward(self, x, context=None, reverse=False):
        n_samples, n_features = x.size()

        # create image
        x = self.image_layer(x)

        ldj = torch.zeros(n_samples, dtype=x.dtype)

        w = self.V
        d_ldj = self.H * self.W * torch.slogdet(w)[1]

        if not reverse:
            w = w.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(x, w, bias=None, stride=1, padding=0, dilation=1, groups=1)

            ldj += d_ldj

            z = self.flatten_layer(z)
            return z, ldj
        else:
            w_inv = torch.inverse(w)
            w_inv = w_inv.view(self.n_channels, self.n_channels, 1, 1)

            z = F.conv2d(x, w_inv, bias=None, stride=1, padding=0, dilation=1, groups=1)

            ldj -= d_ldj

            z = self.flatten_layer(z)

            return z, ldj

    def inverse(self, z, context=None):
        return self(z, context=context, reverse=True)
