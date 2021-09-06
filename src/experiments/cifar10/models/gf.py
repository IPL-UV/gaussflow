import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".here"])

sys.path.append(str(root))

from argparse import ArgumentParser
from typing import Tuple

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from src.models.layers.multiscale import GeneralizedSplitPrior
from src.models.layers.dequantization import UniformDequantization
from src.models.layers.convolutions import Conv1x1Householder
from src.models.layers.mixtures import GaussianMixtureCDF
from src.models.layers.nonlinear import InverseGaussCDF
from src.experiments.utils import gf_propagate
from src.models.gaussianization import (
    init_gaussianization_image_flow,
    init_gaussianization_flow,
)
import torch
import torch.nn as nn
import numpy as np
from src.models.flows.rnvp import append_realnvp_coupling_block_image
from src.models.layers.multiscale import GeneralizedSplitPrior
import torch.distributions as dist
from tqdm import trange


def create_multiscale_cifar10_model_gf_simple(
    X_init: torch.Tensor,
    n_subflows_1: int = 10,
    n_subflows_2: int = 10,
    n_subflows_3: int = 10,
    n_subflows_4: int = 10,
    n_subflows_5: int = 10,
    n_reflections: int = 10,
    n_components: int = 10,
    non_linear: str = "gaussian",
):

    # for the initialization
    init_X = X_init.detach().clone()

    # a simple chain of operations is collected by ReversibleSequential
    inn = Ff.SequenceINN(3, 32, 32)

    # uniform dequantization (for integers)
    inn.append(UniformDequantization, num_bits=8)
    init_X = gf_propagate(inn, init_X)

    print("Input (Full Resolution):")
    n_features, height, width = init_X.shape[1:]
    print(
        f"Dims: {n_features} x {height} x {width} | Total Dims: {n_features*height*width}"
    )

    for _ in trange(n_subflows_1):

        inn, init_X = init_gaussianization_image_flow(
            inn=inn,
            init_X=init_X,
            n_reflections=n_reflections,
            n_components=n_components,
            non_linear=non_linear,
        )

    print("Input (Lower Resolution):")
    # print("DownSampling")
    inn.append(
        Fm.IRevNetDownsampling,
    )
    init_X = gf_propagate(inn, init_X)
    n_channels, height, width = init_X.shape[1:]
    print(
        f"Dims: {n_channels} x {height} x {width} | Total Dims: {n_channels*height*width}"
    )

    for _ in trange(n_subflows_2):

        inn, init_X = init_gaussianization_image_flow(
            inn=inn,
            init_X=init_X,
            n_reflections=n_reflections,
            n_components=n_components,
            non_linear=non_linear,
        )

    print("Input (Lower Resolution):")
    # print("DownSampling")
    inn.append(
        Fm.IRevNetDownsampling,
    )
    init_X = gf_propagate(inn, init_X)
    n_channels, height, width = init_X.shape[1:]
    print(
        f"Dims: {n_channels} x {height} x {width} | Total Dims: {n_channels*height*width}"
    )

    for _ in trange(n_subflows_3):

        inn, init_X = init_gaussianization_image_flow(
            inn=inn,
            init_X=init_X,
            n_reflections=n_reflections,
            n_components=n_components,
            non_linear=non_linear,
        )

    print("Split (1/4):")

    inn.append(
        GeneralizedSplitPrior,
        split=n_channels // 4,
        split_dim=0,
        prior=dist.Normal(0.0, 1.0),
    )

    init_X = gf_propagate(inn, init_X)
    n_channels, height, width = init_X.shape[1:]
    print(
        f"Dims: {n_channels} x {height} x {width} | Total Dims: {n_channels*height*width}"
    )

    print("Input (Lower Resolution):")
    # print("DownSampling")
    inn.append(
        Fm.IRevNetDownsampling,
    )
    init_X = gf_propagate(inn, init_X)
    n_channels, height, width = init_X.shape[1:]
    print(
        f"Dims: {n_channels} x {height} x {width} | Total Dims: {n_channels*height*width}"
    )

    for _ in trange(n_subflows_4):

        inn, init_X = init_gaussianization_image_flow(
            inn=inn,
            init_X=init_X,
            n_reflections=n_reflections,
            n_components=n_components,
            non_linear=non_linear,
        )

    print("Split (1/4):")
    inn.append(
        GeneralizedSplitPrior,
        split=n_channels // 4,
        split_dim=0,
        prior=dist.Normal(0.0, 1.0),
    )

    init_X = gf_propagate(inn, init_X)
    n_channels, height, width = init_X.shape[1:]
    print(
        f"Dims: {n_channels} x {height} x {width} | Total Dims: {n_channels*height*width}"
    )

    print("Flatten (Fully Connected):")
    # print("DownSampling")
    inn.append(Fm.Flatten)

    init_X = gf_propagate(inn, init_X)
    n_features = init_X.shape[1:]
    print(f"Dims: {n_features} | Total Dims: {n_features}")

    for _ in trange(n_subflows_5):

        inn, init_X = init_gaussianization_flow(
            inn=inn,
            init_X=init_X,
            n_reflections=n_reflections,
            n_components=n_components,
            non_linear=non_linear,
        )
    return inn


def add_multiscale_cifar10_model_gf_simple_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # loss function
    parser.add_argument("--n_subflows_1", type=int, default=10)
    parser.add_argument("--n_subflows_2", type=int, default=10)
    parser.add_argument("--n_subflows_3", type=int, default=10)
    parser.add_argument("--n_subflows_4", type=int, default=10)
    parser.add_argument("--n_reflections", type=int, default=12)
    parser.add_argument("--n_components", type=int, default=12)
    parser.add_argument("--non_linear", type=str, default="gaussian")
    return parser


def test_gf_models():

    X_init = torch.randn((64, 3, 32, 32))

    n_subflows_1, n_subflows_2, n_subflows_3, n_subflows_4 = 1, 1, 1, 1

    inn = create_multiscale_cifar10_model_gf_simple(
        X_init=X_init,
        n_subflows_1=n_subflows_1,
        n_subflows_2=n_subflows_2,
        n_subflows_3=n_subflows_3,
        n_subflows_4=n_subflows_4,
    )

    with torch.no_grad():
        z, log_jac_det_for = inn.forward(X_init, rev=False)

        x_ori, log_jac_det_rev = inn.forward(z, rev=True)

        assert x_ori.shape == X_init.shape
        assert log_jac_det_for.shape == log_jac_det_rev.shape

    # inn = create_multiscale_cifar10_model_conv1x1(
    #     img_shape=img_shape,
    #     X_init=X_init,
    #     n_subflows_1=n_subflows_1,
    #     n_subflows_2=n_subflows_2,
    #     n_subflows_3=n_subflows_3,
    # )

    # with torch.no_grad():
    #     z, log_jac_det = inn.forward(X_init, rev=False)

    #     x_ori, log_jac_det = inn.forward(z, rev=True)

    #     assert x_ori.shape == X_init.shape


if __name__ == "__main__":
    test_gf_models()
