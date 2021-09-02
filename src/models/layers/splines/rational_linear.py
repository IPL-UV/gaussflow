"""
This code was copied from: 
https://github.com/hmdolatabadi/LRS_NF/blob/master/nde/transforms/splines/rational_linear.py
"""
import torch
from torch.nn import functional as F
import torch.nn as nn
from nflows.utils import sum_except_batch, searchsorted
import numpy as np

from nflows.transforms import Transform
from src.models.layers.utils import InputOutsideDomain, share_across_batch

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


class PiecewiseRationalLinearCDF(Transform):
    def __init__(
        self,
        shape,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        identity_init=False,
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
    ):
        super().__init__()

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        self.tail_bound = tail_bound
        self.tails = tails

        if identity_init:
            self.unnormalized_widths = nn.Parameter(torch.zeros(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.zeros(*shape, num_bins))
            self.unnormalized_lambdas = nn.Parameter(torch.zeros(*shape, num_bins))

            constant = np.log(np.exp(1 - min_derivative) - 1)
            num_derivatives = (
                (num_bins - 1) if self.tails == "linear" else (num_bins + 1)
            )
            self.unnormalized_derivatives = nn.Parameter(
                constant * torch.ones(*shape, num_derivatives)
            )

        else:
            self.unnormalized_widths = nn.Parameter(torch.rand(*shape, num_bins))
            self.unnormalized_heights = nn.Parameter(torch.rand(*shape, num_bins))
            self.unnormalized_lambdas = nn.Parameter(torch.rand(*shape, num_bins))

            num_derivatives = (
                (num_bins - 1) if self.tails == "linear" else (num_bins + 1)
            )
            self.unnormalized_derivatives = nn.Parameter(
                torch.rand(*shape, num_derivatives)
            )

    def _spline(self, inputs, inverse=False):
        batch_size = inputs.shape[0]

        unnormalized_widths = share_across_batch(self.unnormalized_widths, batch_size)
        unnormalized_heights = share_across_batch(self.unnormalized_heights, batch_size)
        unnormalized_lambdas = share_across_batch(self.unnormalized_lambdas, batch_size)
        unnormalized_derivatives = share_across_batch(
            self.unnormalized_derivatives, batch_size
        )

        if self.tails is None:
            spline_fn = rational_linear_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_linear_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            unnormalized_lambdas=unnormalized_lambdas,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        return self._spline(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        return self._spline(inputs, inverse=True)


def unconstrained_rational_linear_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    unnormalized_lambdas,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):

    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    (
        outputs[inside_interval_mask],
        logabsdet[inside_interval_mask],
    ) = rational_linear_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        unnormalized_lambdas=unnormalized_lambdas[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet


def rational_linear_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    unnormalized_lambdas,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):

    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise InputOutsideDomain()

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left

    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    lambdas = 0.95 * torch.sigmoid(unnormalized_lambdas) + 0.025

    lam = lambdas.gather(-1, bin_idx)[..., 0]
    wa = 1
    wb = torch.sqrt(input_derivatives / input_derivatives_plus_one) * wa
    wc = (
        lam * wa * input_derivatives + (1 - lam) * wb * input_derivatives_plus_one
    ) / input_delta
    ya = input_cumheights
    yb = input_heights + input_cumheights
    yc = ((1 - lam) * wa * ya + lam * wb * yb) / ((1 - lam) * wa + lam * wb)

    if inverse:

        numerator = (lam * wa * (ya - inputs)) * (inputs <= yc).float() + (
            (wc - lam * wb) * inputs + lam * wb * yb - wc * yc
        ) * (inputs > yc).float()

        denominator = ((wc - wa) * inputs + wa * ya - wc * yc) * (
            inputs <= yc
        ).float() + ((wc - wb) * inputs + wb * yb - wc * yc) * (inputs > yc).float()

        theta = numerator / denominator

        outputs = theta * input_bin_widths + input_cumwidths

        derivative_numerator = (
            wa * wc * lam * (yc - ya) * (inputs <= yc).float()
            + wb * wc * (1 - lam) * (yb - yc) * (inputs > yc).float()
        ) * input_bin_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(abs(denominator))

        return outputs, logabsdet
    else:

        theta = (inputs - input_cumwidths) / input_bin_widths

        numerator = (wa * ya * (lam - theta) + wc * yc * theta) * (
            theta <= lam
        ).float() + (wc * yc * (1 - theta) + wb * yb * (theta - lam)) * (
            theta > lam
        ).float()

        denominator = (wa * (lam - theta) + wc * theta) * (theta <= lam).float() + (
            wc * (1 - theta) + wb * (theta - lam)
        ) * (theta > lam).float()

        outputs = numerator / denominator

        derivative_numerator = (
            wa * wc * lam * (yc - ya) * (theta <= lam).float()
            + wb * wc * (1 - lam) * (yb - yc) * (theta > lam).float()
        ) / input_bin_widths

        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(abs(denominator))

        return outputs, logabsdet