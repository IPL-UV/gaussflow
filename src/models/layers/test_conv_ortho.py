from typing import Mapping
import FrEIA.framework as Ff
import numpy as np
import torch

from src.models.layers.conv import Conv1x1Householder


def test_ortho_conv1x1_shape():

    # create fake data
    n_batch = 128
    n_channels = 3
    H = 32
    W = 32
    n_reflections = 10

    x = np.random.randn(n_batch, n_channels, H, W)
    dims_in = [
        (
            n_channels,
            H,
            W,
        )
    ]

    # do transformation
    with torch.no_grad():

        conv_layer = Conv1x1Householder(dims_in=dims_in, n_reflections=n_reflections)

        z, log_abs_det = conv_layer.forward([torch.Tensor(x)])

        assert z[0].shape == x.shape
        assert log_abs_det.shape[0] == x.shape[0]

        x_recon, log_abs_det_r = conv_layer.forward(z, rev=True)

        assert x_recon[0].shape == x.shape
        assert log_abs_det_r.shape[0] == x.shape[0]

        diff = torch.mean((torch.Tensor(x) - x_recon[0]) ** 2)

        assert diff < 1e-10, f"Diff '{diff}' is larger than 1e-10"

    return None


def test_ortho_conv1x1_inn():

    # create fake data
    n_batch = 128
    n_channels = 3
    H = 32
    W = 32
    n_reflections = 10

    x = np.random.randn(n_batch, n_channels, H, W)
    n_features = (n_channels, H, W)

    # do transformation
    with torch.no_grad():

        # initialize sequence
        inn = Ff.SequenceINN(*n_features)

        # append layer
        inn.append(Conv1x1Householder, n_reflections=n_reflections)

        z, log_abs_det = inn(torch.Tensor(x))

        assert z.shape == x.shape
        assert log_abs_det.shape[0] == x.shape[0]

        x_recon, log_abs_det_r = inn(z, rev=True)

        assert x_recon.shape == x.shape
        assert log_abs_det_r.shape[0] == x.shape[0]

        diff = torch.mean((torch.Tensor(x) - x_recon) ** 2)

        assert diff < 1e-10, f"Diff '{diff}' is larger than 1e-10"

    return None


if __name__ == "__main__":
    # test_ortho_conv1x1_shape()
    test_ortho_conv1x1_inn()