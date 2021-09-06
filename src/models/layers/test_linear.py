from typing import Mapping
import FrEIA.framework as Ff
import numpy as np
import torch

from src.models.layers.linear import create_linear_transform, LinearLayer

TRANSFORMS = ["householder", "svd", "permutation", "lu"]


def test_householder_shape():

    # create 2D Data
    n_batch = 128
    n_features = 10
    n_reflections = 10

    for transform in TRANSFORMS:

        x = np.random.randn(n_batch, n_features)
        dims_in = [(n_features,)]

        # do transformation
        with torch.no_grad():

            hh_layer = LinearLayer(
                dims_in=dims_in, transform=transform, n_reflections=n_reflections
            )

            z, log_abs_det = hh_layer.forward([torch.Tensor(x)])

            assert z[0].shape == x.shape
            assert log_abs_det.shape[0] == x.shape[0]

            x_recon, log_abs_det_r = hh_layer.forward(z, rev=True)

            assert x_recon[0].shape == x.shape
            assert log_abs_det_r.shape[0] == x.shape[0]

            diff = torch.mean((torch.Tensor(x) - x_recon[0]) ** 2)

            assert diff < 1e-10, f"Diff '{diff}' is larger than 1e-10"

    return None


def test_householder_inn():

    # create 2D Data
    n_batch = 128
    n_features = 10
    n_reflections = 10
    transform = "householder"

    x = np.random.randn(n_batch, n_features)

    for transform in TRANSFORMS:

        # do transformation
        with torch.no_grad():

            # initialize sequence
            inn = Ff.SequenceINN(n_features)

            # append layer
            inn.append(LinearLayer, transform=transform, n_reflections=n_reflections)

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
    test_householder_shape()
    test_householder_inn()
