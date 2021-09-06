from argparse import ArgumentParser
from typing import Tuple

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from src.models.layers.dequantization import UniformDequantization
from src.models.layers.convolutions import Conv1x1, Conv1x1Householder, ConvExponential
import torch
import torch.nn as nn
import numpy as np
from src.models.flows.rnvp import append_realnvp_coupling_block_image
from src.models.layers.multiscale import GeneralizedSplitPrior
import torch.distributions as dist


def create_multiscale_cifar10_model_permute(
    img_shape: Tuple,
    X_init: torch.Tensor,
    n_subflows_1: int = 4,
    n_subflows_2: int = 4,
    n_subflows_3: int = 4,
):

    n_channels, height, width = img_shape

    # subset net
    def subnet_conv(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, c_out, 3, padding=1),
        )

    inn = Ff.SequenceINN(n_channels, height, width)

    # uniform dequantization (for integers)
    inn.append(UniformDequantization, num_bits=8)

    print("Input (Higher Resolution):")
    n_features, height, width = inn(X_init)[0].shape[1:]
    print(
        f"Dims: {n_features} x {height} x {width} | Total Dims: {n_features*height*width}"
    )

    # HIGHER RESOLUTION SECTION
    for ilayer in range(n_subflows_1):

        # append RealNVP Coupling Block
        inn.append(
            Fm.GLOWCouplingBlock,
            subnet_constructor=subnet_conv,
            clamp=1.2,
            # name=f"glow_high_res_{ilayer}",
        )
        inn.append(Fm.PermuteRandom, seed=ilayer)

    # LOWER RESOLUTION SECTION
    print("Lower Resolution")
    inn.append(
        Fm.IRevNetDownsampling,
    )

    n_features, height, width = inn(X_init)[0].shape[1:]
    print(
        f"Dims: {n_features} x {height} x {width} | Total Dims: {n_features*height*width}"
    )

    # subset net
    def subnet_conv(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, c_out, 3, padding=1),
        )

    def subnet_conv_1x1(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, c_out, 1),
        )

    for ilayer in range(n_subflows_2):

        if ilayer % 2 == 0:
            conditioner = subnet_conv_1x1
        else:
            conditioner = subnet_conv

        inn.append(
            Fm.GLOWCouplingBlock,
            subnet_constructor=conditioner,
            clamp=1.2,
        )
        inn.append(Fm.PermuteRandom, seed=ilayer)

    # FULLY CONNECTED
    print("Flatten:")
    inn.append(Fm.Flatten)

    n_features = int(inn(X_init)[0].shape[1:][0])
    print(f"Dims: {n_features} | Total Dims: {n_features}")

    # Split Outputs

    print("Split:")
    inn.append(
        GeneralizedSplitPrior,
        split=n_features // 4,
        split_dim=0,
        prior=dist.Normal(0.0, 1.0),
    )

    n_features = int(inn(X_init)[0].shape[1:][0])
    print(f"Dims: {n_features} | Total Dims: {n_features}")

    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, 512), nn.ReLU(), nn.Linear(512, dims_out)
        )

    for ilayer in range(n_subflows_3):

        inn.append(
            Fm.GLOWCouplingBlock,
            subnet_constructor=subnet_fc,
            clamp=1.2,
        )
        inn.append(Fm.PermuteRandom, seed=ilayer)

    print("Final:")
    n_features = int(inn(X_init)[0].shape[1:][0])
    print(f"Dims: {n_features} | Total Dims: {n_features}")
    print("Done!")
    return inn


def create_multiscale_cifar10_model_conv1x1(
    img_shape: Tuple,
    X_init: torch.Tensor,
    n_subflows_1: int = 4,
    n_subflows_2: int = 12,
    n_subflows_3: int = 12,
    n_reflections: int = 10,
):

    n_channels, height, width = img_shape

    # subset net
    def subnet_conv(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, c_out, 3, padding=1),
        )

    inn = Ff.SequenceINN(n_channels, height, width)

    # uniform dequantization (for integers)
    inn.append(UniformDequantization, num_bits=8)

    print("Input (Higher Resolution):")
    n_features, height, width = inn(X_init)[0].shape[1:]
    print(
        f"Dims: {n_features} x {height} x {width} | Total Dims: {n_features*height*width}"
    )

    # HIGHER RESOLUTION SECTION
    for ilayer in range(n_subflows_1):

        # append RealNVP Coupling Block
        inn.append(
            Fm.GLOWCouplingBlock,
            subnet_constructor=subnet_conv,
            clamp=1.2,
            # name=f"glow_high_res_{ilayer}",
        )
        inn.append(
            Fm.ActNorm,
        )
        inn.append(
            Conv1x1Householder,
            n_reflections=n_reflections,
        )

    # LOWER RESOLUTION SECTION
    print("Lower Resolution")
    inn.append(
        Fm.IRevNetDownsampling,
    )

    n_features, height, width = inn(X_init)[0].shape[1:]
    print(
        f"Dims: {n_features} x {height} x {width} | Total Dims: {n_features*height*width}"
    )

    # subset net
    def subnet_conv(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, c_out, 3, padding=1),
        )

    def subnet_conv_1x1(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, c_out, 1),
        )

    for ilayer in range(n_subflows_2):

        if ilayer % 2 == 0:
            conditioner = subnet_conv_1x1
        else:
            conditioner = subnet_conv

        inn.append(
            Fm.GLOWCouplingBlock,
            subnet_constructor=conditioner,
            clamp=1.2,
        )
        inn.append(
            Fm.ActNorm,
        )
        inn.append(
            Conv1x1Householder,
            n_reflections=n_reflections,
        )

    # FULLY CONNECTED
    print("Flatten:")
    inn.append(Fm.Flatten)

    n_features = int(inn(X_init)[0].shape[1:][0])
    print(f"Dims: {n_features} | Total Dims: {n_features}")

    # Split Outputs

    print("Split:")
    inn.append(
        GeneralizedSplitPrior,
        split=n_features // 4,
        split_dim=0,
        prior=dist.Normal(0.0, 1.0),
    )

    n_features = int(inn(X_init)[0].shape[1:][0])
    print(f"Dims: {n_features} | Total Dims: {n_features}")

    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(
            nn.Linear(dims_in, 512), nn.ReLU(), nn.Linear(512, dims_out)
        )

    for ilayer in range(n_subflows_3):

        inn.append(
            Fm.GLOWCouplingBlock,
            subnet_constructor=subnet_fc,
            clamp=1.2,
        )
        inn.append(
            Fm.ActNorm,
        )
        inn.append(
            Fm.HouseholderPerm,
            n_reflections=n_reflections,
        )

    print("Final:")
    n_features = int(inn(X_init)[0].shape[1:][0])
    print(f"Dims: {n_features} | Total Dims: {n_features}")
    print("Done!")
    return inn


def add_multiscale_cifar10_model_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # loss function
    parser.add_argument("--n_subflows_1", type=int, default=2)
    parser.add_argument("--n_subflows_2", type=int, default=12)
    parser.add_argument("--n_subflows_3", type=int, default=12)
    parser.add_argument("--actnorm", type=bool, default=True)
    parser.add_argument("--n_reflections", type=int, default=2)
    return parser


def test_glow_models():

    img_shape = (3, 32, 32)
    X_init = torch.randn((64, 3, 32, 32))

    n_subflows_1, n_subflows_2, n_subflows_3 = 4, 12, 12

    inn = create_multiscale_cifar10_model_permute(
        img_shape=img_shape,
        X_init=X_init,
        n_subflows_1=n_subflows_1,
        n_subflows_2=n_subflows_2,
        n_subflows_3=n_subflows_3,
    )

    with torch.no_grad():
        z, log_jac_det = inn.forward(X_init, rev=False)

        x_ori, log_jac_det = inn.forward(z, rev=True)

        assert x_ori.shape == X_init.shape

    inn = create_multiscale_cifar10_model_conv1x1(
        img_shape=img_shape,
        X_init=X_init,
        n_subflows_1=n_subflows_1,
        n_subflows_2=n_subflows_2,
        n_subflows_3=n_subflows_3,
    )

    with torch.no_grad():
        z, log_jac_det = inn.forward(X_init, rev=False)

        x_ori, log_jac_det = inn.forward(z, rev=True)

        assert x_ori.shape == X_init.shape


if __name__ == "__main__":
    test_glow_models()
