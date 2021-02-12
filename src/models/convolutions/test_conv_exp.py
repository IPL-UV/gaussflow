from typing import Mapping

import numpy as np
import torch

from src.models.convolutions.conv_exp import ConvExp


def test_conv_exp_shape():

    # create fake data
    n_batch = 128
    n_channels = 3
    H = 3
    W = 3
    n_reflections = 10

    x = np.random.randn(n_batch, n_channels, H, W)

    # convert to flattened image
    x = np.reshape(x, (n_batch, H * W * n_channels))

    # do transformation
    with torch.no_grad():

        conv_layer = ConvExp(
            n_reflections=n_reflections, n_channels=n_channels, H=H, W=W
        )
        z, log_abs_det = conv_layer.forward(torch.Tensor(x))

        assert z.shape == x.shape
        assert log_abs_det.shape[0] == x.shape[0]

        x_recon, log_abs_det_r = conv_layer.inverse(z)

        assert x_recon.shape == x.shape
        assert log_abs_det_r.shape[0] == x.shape[0]

        diff = torch.mean((torch.Tensor(x) - x_recon) ** 2)

        assert diff < 1e-10, f"Diff '{diff}' is larger than 1e-10"

    return None


if __name__ == "__main__":
    test_conv_exp_shape()