from typing import Iterable, List, Optional, Callable
import FrEIA.modules as Fm
from nflows.transforms import PiecewiseRationalQuadraticCDF
from nflows.transforms.splines.rational_quadratic import (
    DEFAULT_MIN_BIN_HEIGHT,
    DEFAULT_MIN_BIN_WIDTH,
    DEFAULT_MIN_DERIVATIVE,
)
from src.models.layers.splines.rational_linear import PiecewiseRationalLinearCDF
from nflows.utils import sum_except_batch


class RationalQuadraticSplines(Fm.InvertibleModule):
    def __init__(
        self,
        dims_in: Iterable[List[int]],
        num_bins: int = 10,
        tails: Optional[str] = None,
        tail_bound: float = 5.0,
        identity_init: bool = False,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
    ):
        super().__init__(dims_in)

        self.transform = PiecewiseRationalQuadraticCDF(
            shape=dims_in[0],
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            identity_init=identity_init,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )

    def forward(self, x, rev=False, jac=True):

        # x is passed to the function as a list (in this case of only on element)
        x = x[0]
        if not rev:
            # forward operation
            x, log_jac_det = self.transform.forward(x)
        else:
            # backward operation
            x, log_jac_det = self.transform.inverse(x)

        log_jac_det = sum_except_batch(log_jac_det)

        return (x,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims


class RationalLinearSplines(RationalQuadraticSplines):
    def __init__(
        self,
        dims_in: Iterable[List[int]],
        num_bins: int = 10,
        tails: Optional[str] = None,
        tail_bound: float = 5.0,
        identity_init: bool = False,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
    ):
        super().__init__(dims_in)

        self.transform = PiecewiseRationalLinearCDF(
            shape=dims_in,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            identity_init=identity_init,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )
