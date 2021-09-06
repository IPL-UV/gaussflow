from typing import Callable

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from src.models.layers.dequantization import UniformDequantization
from src.models.layers.convolutions import Conv1x1, Conv1x1Householder, ConvExponential


def append_realnvp_coupling_block_image(
    inn,
    conditioner: Callable,
    n_reflections=10,
    actnorm: bool = False,
    mask: str = "checkerboard",
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
    else:
        raise ValueError(f"Unrecognized masking method: {mask}")

    # =================
    # RealNVP Coupling
    # =================
    inn.append(
        Fm.RNVPCouplingBlock,
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
    else:
        raise ValueError(f"Unrecognized masking method: {mask}")

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
    if n_reflections is not None:
        inn.append(
            Conv1x1Householder,
            n_reflections=n_reflections,
        )
    else:
        inn.append(Conv1x1)
    return inn
