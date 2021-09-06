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


def create_simple_mnist_model(
    img_shape: Tuple,
    X_init: torch.Tensor,
    n_subflows: int = 8,
    actnorm: bool = True,
    n_reflections: int = 2,
    mask: str = "checkerboard",
):

    n_channels, height, width = img_shape

    # subset net
    def subnet_conv(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, c_out, 3, padding=1),
        )

    inn = Ff.SequenceINN(n_channels, height, width)

    # uniform dequantization (for integers)
    inn.append(UniformDequantization, num_bits=8)

    print("Input:")
    print(X_init.shape, np.prod(inn(X_init)[0].shape))

    for _ in range(n_subflows):

        # append RealNVP Coupling Block
        inn = append_realnvp_coupling_block_image(
            inn,
            conditioner=subnet_conv,
            actnorm=actnorm,
            n_reflections=n_reflections,
            mask=mask,
        )

    inn.append(Fm.Flatten)
    print("Final:")
    print(inn(X_init)[0].shape)
    return inn


def add_mnist_model_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # loss function
    parser.add_argument("--n_subflows_1", type=int, default=8)
    parser.add_argument("--n_subflows_2", type=int, default=4)
    parser.add_argument("--actnorm", type=bool, default=True)
    parser.add_argument("--n_reflections", type=int, default=2)
    parser.add_argument("--mask", type=str, default="checkerboard")
    return parser


def create_multiscale_mnist_model(
    img_shape: Tuple,
    X_init: torch.Tensor,
    n_subflows_1: int = 2,
    n_subflows_2: int = 4,
    actnorm: bool = True,
    n_reflections: int = 2,
    mask: str = "checkerboard",
):

    n_channels, height, width = img_shape

    # subset net
    def subnet_conv(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, c_out, 3, padding=1),
        )

    inn = Ff.SequenceINN(n_channels, height, width)

    # uniform dequantization (for integers)
    inn.append(UniformDequantization, num_bits=8)

    print("Input:")
    print(X_init.shape, np.prod(inn(X_init)[0].shape))

    for _ in range(n_subflows_1):

        # append RealNVP Coupling Block
        inn = append_realnvp_coupling_block_image(
            inn,
            conditioner=subnet_conv,
            actnorm=actnorm,
            n_reflections=n_reflections,
            mask=mask,
        )

    # SCALE I
    print(f"Scale: 1")
    print("DownSample")
    inn.append(
        Fm.IRevNetDownsampling,
    )

    print(inn(X_init)[0].shape, np.prod(inn(X_init)[0].shape))

    # subset net
    def subnet_conv(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, c_out, 3, padding=1),
        )

    for _ in range(n_subflows_2):

        # append RealNVP Coupling Block
        inn = append_realnvp_coupling_block_image(
            inn,
            conditioner=subnet_conv,
            actnorm=actnorm,
            n_reflections=n_reflections,
            mask=mask,
        )

    inn.append(Fm.Flatten)
    print("Final:")
    print(inn(X_init)[0].shape)
    return inn


def add_multiscale_mnist_model_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # loss function
    parser.add_argument("--n_subflows_1", type=int, default=2)
    parser.add_argument("--n_subflows_2", type=int, default=4)
    parser.add_argument("--actnorm", type=bool, default=True)
    parser.add_argument("--n_reflections", type=int, default=2)
    parser.add_argument("--mask", type=str, default="checkerboard")
    return parser
