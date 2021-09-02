from src.models.layers.mixtures import MixtureGaussianCDFCoupling
from FrEIA.modules import coupling_layers
from src.models.layers.splines import (
    RationalLinearSplineCouplingBlock,
    RationalQuadraticSplineCouplingBlock,
)
from typing import Callable, Union
import FrEIA.modules as Fm


class FlowCouplingBlock:
    def __init__(
        self,
        dims_in,
        dims_c=[],
        subnet_constructor: Callable = None,
        bijector: Callable = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "ATAN",
    ):
        super().__init__(dims_in, dims_c, clamp, clamp_activation)

    def _coupling1(self, x1, u2, rev=False):
        pass

    def _coupling2(self, x2, u1, rev=False):
        pass


def get_coupling_layer(coupling: str = "glow"):
    # coupling transform (GLOW)
    if coupling == "nice":
        coupling_transform = Fm.NICECouplingBlock

    elif coupling == "realnvp":
        coupling_transform = Fm.RNVPCouplingBlock

    elif coupling == "glow":
        coupling_transform = Fm.GLOWCouplingBlock

    elif coupling == "rqs":
        coupling_transform = RationalQuadraticSplineCouplingBlock

    elif coupling == "rls":
        coupling_transform = RationalLinearSplineCouplingBlock

    elif coupling == "flowpp":
        coupling_transform = MixtureGaussianCDFCoupling

    else:
        raise ValueError(f"unrecognized coupling transform: {coupling}")

    return coupling_transform
