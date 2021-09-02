from typing import Mapping
import FrEIA.framework as Ff
import numpy as np
import torch
import torch.nn as nn

from src.models.layers.mixtures import MixtureGaussianCDFCoupling
from src.models.nets.resnet import ResidualNet


def nflows_resnet(dims_in, dims_out):
    return ResidualNet(
        dims_in, dims_out, hidden_features=32, num_blocks=2, activation=nn.ReLU()
    )


def test_mixture_gauss_cdf_coupling_shape():

    # create fake data
    n_batch = 128
    n_features = 10
    n_components = 10

    x = np.random.randn(n_batch, n_features)
    dims_in = [(n_features,)]

    # do transformation
    with torch.no_grad():

        conv_layer = MixtureGaussianCDFCoupling(
            dims_in=dims_in, n_components=n_components, subnet_constructor=nflows_resnet
        )

        z, log_abs_det = conv_layer.forward([torch.Tensor(x)])

        assert z[0].shape == x.shape
        assert log_abs_det.shape[0] == x.shape[0]

        x_recon, log_abs_det_r = conv_layer.forward(z, rev=True)

        assert x_recon[0].shape == x.shape
        assert log_abs_det_r.shape[0] == x.shape[0]

        diff = torch.mean((torch.Tensor(x) - x_recon[0]) ** 2)

        assert diff < 1e-10, f"Diff '{diff}' is larger than 1e-10"

    return None


# def test_ortho_conv1x1_inn():

#     # create fake data
#     n_batch = 128
#     n_channels = 3
#     H = 32
#     W = 32
#     n_reflections = 10

#     x = np.random.randn(n_batch, n_channels, H, W)
#     n_features = (n_channels, H, W)

#     # do transformation
#     with torch.no_grad():

#         # initialize sequence
#         inn = Ff.SequenceINN(*n_features)

#         # append layer
#         inn.append(Conv1x1Householder, n_reflections=n_reflections)

#         z, log_abs_det = inn(torch.Tensor(x))

#         assert z.shape == x.shape
#         assert log_abs_det.shape[0] == x.shape[0]

#         x_recon, log_abs_det_r = inn(z, rev=True)

#         assert x_recon.shape == x.shape
#         assert log_abs_det_r.shape[0] == x.shape[0]

#         diff = torch.mean((torch.Tensor(x) - x_recon) ** 2)

#         assert diff < 1e-10, f"Diff '{diff}' is larger than 1e-10"

#     return None


if __name__ == "__main__":
    test_mixture_gauss_cdf_coupling_shape()
    # test_ortho_conv1x1_inn()