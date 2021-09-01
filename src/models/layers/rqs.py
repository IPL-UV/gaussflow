from typing import List, Optional, Callable
import warnings
import FrEIA.modules as Fm
from FrEIA.modules.coupling_layers import _BaseCouplingBlock
from nflows import transforms
from nflows.transforms import splines
from nflows.transforms.splines.rational_quadratic import (
    DEFAULT_MIN_BIN_HEIGHT,
    DEFAULT_MIN_BIN_WIDTH,
    DEFAULT_MIN_DERIVATIVE,
)
import torch


class RQSCouplingBlock(Fm.InvertibleModule):
    def __init__(
        self,
        dims_in: List[int],
        mask: List[int],
        subnet_constructor: Callable,
        num_bins: int = 10,
        tails: Optional[str] = None,
        tail_bound: float = 1.0,
        **kwargs
    ):
        super().__init__(dims_in)

        self.transform = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=subnet_constructor,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            **kwargs
        )

    def forward(self, x, c=[], rev=False, jac=True):
        # the Jacobian term is trivial to calculate so we return it
        # even if jac=False

        # x is passed to the function as a list (in this case of only on element)
        x = x[0]
        if not rev:

            # forward operation
            z, log_jac_det = self.transform.forward(x, context=None)
        else:
            # backward operation
            z, log_jac_det = self.transform.inverse(x, context=None)

        return (z,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims


class RQSplines(Fm.InvertibleModule):
    def __init__(
        self,
        dims_in: List[int],
        num_bins: int = 10,
        tails: Optional[str] = None,
        tail_bound: float = 1.0,
        identity_init: bool = False,
    ):
        super().__init__(dims_in)

        self.transform = transforms.PiecewiseRationalQuadraticCDF(
            shape=dims_in[0],
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            identity_init=identity_init,
        )

    def forward(self, x, rev=False, jac=True):
        # the Jacobian term is trivial to calculate so we return it
        # even if jac=False

        # x is passed to the function as a list (in this case of only on element)
        x = x[0]
        if not rev:
            # forward operation
            x, log_jac_det = self.transform.forward(x)
        else:
            # backward operation
            x, log_jac_det = self.transform.inverse(x)

        return (x,), log_jac_det

    def output_dims(self, input_dims):
        return input_dims


class CustomRQSBlock(_BaseCouplingBlock):
    def __init__(
        self,
        dims_in: List[int],
        mask: List[int],
        subnet_constructor: Callable,
        dims_c=[],
        num_bins: int = 10,
        tails: Optional[str] = None,
        tail_bound: float = 1.0,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
        **kwargs
    ):
        super().__init__(dims_in, dims_c, clamp=2.0, clamp_activation="ATAN")

        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.transform_net = None

        self.subnet1 = subnet_constructor(
            self.split_len1 + self.condition_length, self.split_len2 * 2
        )
        self.subnet2 = subnet_constructor(
            self.split_len2 + self.condition_length, self.split_len1 * 2
        )

        self.transform = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=subnet_constructor,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            **kwargs
        )

        if self.tails is None:
            self.spline_fn = splines.rational_quadratic_spline
            self.spline_kwargs = {}
        else:
            self.spline_fn = splines.unconstrained_rational_quadratic_spline
            self.spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

    def forward(self, x, rev=False, jac=True):
        # the Jacobian term is trivial to calculate so we return it
        # even if jac=False

        # x is passed to the function as a list (in this case of only on element)
        x = x[0]
        if not rev:
            # forward operation
            x, log_jac_det = self.transform.forward(x)
        else:
            # backward operation
            x, log_jac_det = self.transform.inverse(x)

        return (x,), log_jac_det

    def _piecewise_cdf(self, inputs, params, rev=False):

        # extract parameters
        unnormalized_widths = params[..., : self.num_bins]
        unnormalized_heights = params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = params[..., 2 * self.num_bins :]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= torch.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= torch.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= torch.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= torch.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn(
                "Inputs to the softmax are not scaled down: initialization might be bad."
            )

        return self.spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=rev,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **self.spline_kwargs
        )

    def _coupling1(self, x1, u2, rev=False):

        return self._piecewise_cdf(x1, u2, inverse=rev)

    def _coupling2(self, x2, u1, rev=False):
        return self._piecewise_cdf(x2, u1, inverse=rev)

    def output_dims(self, input_dims):
        return input_dims
