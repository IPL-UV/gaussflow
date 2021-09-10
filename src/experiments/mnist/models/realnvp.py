import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
from src.models.layers.dequantization import UniformDequantization
from src.models.flows.realnvp import append_realnvp_coupling_block_image
from src.models.layers.multiscale import SplitPrior
import torch.distributions as dist


def get_multiscale_realnvp_model(
    X_init: torch.Tensor, mask: str = "checkerboard", quantization: bool = True
):

    # a simple chain of operations is collected by ReversibleSequential

    n_channels = 1
    height = 28
    width = 28

    inn = Ff.SequenceINN(n_channels, height, width)

    # uniform dequantization (for integers)
    if quantization:
        inn.append(UniformDequantization, num_bits=8)

    print("Input:")
    print(X_init.shape, np.prod(inn(X_init)[0].shape))

    # subset net
    def subnet_conv(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, c_out, 3, padding=1),
        )

    for _ in range(2):

        # append RealNVP Coupling Block
        inn = append_realnvp_coupling_block_image(
            inn, conditioner=subnet_conv, mask=mask, actnorm=False, permute=True
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
            nn.Conv2d(c_in, 48, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, c_out, 3, padding=1),
        )

    for _ in range(2):

        # append RealNVP Coupling Block
        inn = append_realnvp_coupling_block_image(
            inn, conditioner=subnet_conv, mask=mask, actnorm=False, permute=True
        )

    print("Split I")
    inn.append(SplitPrior, prior=dist.Normal(0.0, 1.0))
    print(inn(X_init)[0].shape, np.prod(inn(X_init)[0].shape))

    # SCALE I
    print(f"Scale II")
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

    for _ in range(4):

        # append RealNVP Coupling Block
        inn = append_realnvp_coupling_block_image(
            inn,
            conditioner=subnet_conv,
            actnorm=False,
            mask=None,
            permute=True,
        )

    inn.append(Fm.Flatten)
    print("Final:")
    output_shape = list(inn(X_init)[0].shape)[1]
    print(output_shape)

    return inn, output_shape
