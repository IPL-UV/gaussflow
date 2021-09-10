from typing import Callable

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from src.models.layers.dequantization import UniformDequantization
from src.models.layers.convolutions import Conv1x1, Conv1x1Householder, ConvExponential


def append_glow_coupling_block_image(
    inn,
    conditioner: Callable,
    actnorm: bool = False,
    mask: str = "checkerboard",
    permute: bool = True,
    n_reflections=10,
):
    # =======================
    # checkboard downsampling
    # =======================
    if mask == "checkerboard":
        inn.append(
            Fm.IRevNetDownsampling,
        )
    elif mask == "wavelet":
        inn.append(
            Fm.HaarDownsampling,
        )

    # =================
    # RealNVP Coupling
    # =================
    inn.append(
        Fm.GLOWCouplingBlock,
        subnet_constructor=conditioner,
    )
    # Upsampling
    if mask == "checkerboard":
        inn.append(
            Fm.IRevNetUpsampling,
        )
    elif mask == "wavelet":
        inn.append(
            Fm.HaarUpsampling,
        )

    # =========
    # act norm
    # =========
    if actnorm:
        inn.append(
            Fm.ActNorm,
        )
    # ===============
    # 1x1 Convolution with householder parameterization
    # ===============
    if permute:
        inn.append(Fm.PermuteRandom)
    elif n_reflections is not None:
        inn.append(
            Conv1x1Householder,
            n_reflections=n_reflections,
        )
    else:
        inn.append(Conv1x1)
    return inn
