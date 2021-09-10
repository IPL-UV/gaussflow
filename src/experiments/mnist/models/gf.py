import torch
from tqdm import trange
import numpy as np
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from src.models.flows.gaussianization import (
    append_gaussflow_image_block,
    append_gaussflow_tabular_block,
)
from src.experiments.utils import gf_propagate
from src.models.layers.dequantization import UniformDequantization


def get_gf_fc_model(
    X_init: torch.Tensor, n_layers: int = 20, non_linear: str = "gaussian"
):

    # a simple chain of operations is collected by ReversibleSequential

    n_channels = 1
    height = 28
    width = 28

    inn = Ff.SequenceINN(n_channels, height, width)

    init_X = X_init

    print("Input:")
    print(init_X.shape, np.prod(inn(init_X)[0].shape[1:]))

    # uniform dequantization (for integers)
    inn.append(UniformDequantization, num_bits=8)

    init_X = gf_propagate(inn, init_X)

    print(init_X.min(), init_X.max())

    print("Flatten:")
    inn.append(Fm.Flatten)
    init_X = gf_propagate(inn, init_X)
    print(init_X.shape, np.prod(inn(init_X)[0].shape[1:]))

    for _ in trange(n_layers):

        # append RealNVP Coupling Block
        inn, init_X = append_gaussflow_tabular_block(
            inn,
            init_X=init_X,
            non_linear=non_linear,
            n_components=8,
            n_reflections=20,
        )

    print("Final:")
    print(init_X.shape, np.prod(inn(init_X)[0].shape[1:]))

    output_dim = init_X.shape[1]

    return inn, output_dim


def get_gf_fc_spline_model(X_init: torch.Tensor, n_layers: int = 20):

    # a simple chain of operations is collected by ReversibleSequential

    n_channels = 1
    height = 28
    width = 28

    inn = Ff.SequenceINN(n_channels, height, width)

    init_X = X_init

    print("Input:")
    print(init_X.shape, np.prod(inn(init_X)[0].shape[1:]))

    # uniform dequantization (for integers)
    inn.append(UniformDequantization, num_bits=8)

    init_X = gf_propagate(inn, init_X)

    print(init_X.min(), init_X.max())

    print("Flatten:")
    inn.append(Fm.Flatten)
    init_X = gf_propagate(inn, init_X)
    print(init_X.shape, np.prod(inn(init_X)[0].shape[1:]))

    for _ in trange(n_layers):

        # append RealNVP Coupling Block
        inn, init_X = append_gaussflow_tabular_block(
            inn,
            init_X=init_X,
            non_linear="gaussian",
            n_components=8,
            n_reflections=10,
        )

    print("Final:")
    print(init_X.shape, np.prod(inn(init_X)[0].shape[1:]))

    output_dim = init_X.shape[1]

    return inn, output_dim
