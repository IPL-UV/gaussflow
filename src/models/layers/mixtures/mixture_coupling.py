from nflows.utils.torchutils import sum_except_batch
from src.models.layers.mixtures.mixture_gaussian import (
    mixture_gauss_cdf,
    mixture_gauss_log_pdf,
    mixture_inv_cdf,
)
from typing import List, Optional, Callable, NamedTuple
import warnings
import torch
import torch.distributions as dist
import torch.nn.functional as F
import FrEIA.modules as Fm
from FrEIA.modules.coupling_layers import _BaseCouplingBlock
from nflows import transforms
from torch.distributions import Normal
from src.models.layers.utils import bisection_inverse


class MixtureGaussianCDFCoupling(_BaseCouplingBlock):
    def __init__(
        self,
        dims_in: List[int],
        subnet_constructor: Callable,
        dims_c=[],
        n_components: int = 10,
        inv_eps: float = 1e-10,
        catch_error: bool = False,
        **kwargs,
    ):
        super().__init__(dims_in, dims_c, clamp=2.0, clamp_activation="ATAN")
        # dimensionality of neural net
        output_dims = self.split_len2 * 2  # affine coefficients
        self.mixture_coeffs_dim = n_components * self.split_len2
        output_dims += 3 * self.mixture_coeffs_dim  # mixture params
        input_dims = self.split_len1 + self.condition_length

        self.subnet = subnet_constructor(input_dims, output_dims)

        self.inv_eps = inv_eps
        self.n_components = n_components
        self.catch_error = catch_error

    def _unpack_params(self, params):

        # affine coefficients
        affine_params = params[..., : 2 * self.split_len1]

        affine_scale = affine_params[..., : self.split_len1]
        affine_add = affine_params[..., self.split_len1 : 2 * self.split_len1]

        # mixture params
        mixture_params = params[..., 2 * self.split_len1 :]

        weight_logits = mixture_params[..., : self.mixture_coeffs_dim]
        weight_logits = weight_logits.reshape(
            weight_logits.size(0), self.split_len1, self.n_components
        )

        locs = mixture_params[
            ..., self.mixture_coeffs_dim : 2 * self.mixture_coeffs_dim
        ]
        locs = locs.reshape(locs.size(0), self.split_len1, self.n_components)

        log_scales = mixture_params[..., 2 * self.mixture_coeffs_dim :]
        log_scales = log_scales.reshape(
            log_scales.size(0), self.split_len1, self.n_components
        )

        return affine_scale, affine_add, weight_logits, locs, log_scales

    def forward(self, x, c=[], rev=False, jac=True):

        # x is passed to the function as a list (in this case of only on element)
        x = x[0]

        # split data
        x1, x2 = torch.split(x, [self.split_len1, self.split_len2], dim=1)

        # add conditional data (optional)
        x1_c = torch.cat([x1, *c]) if self.conditional else x1

        # transform params
        all_params = self.subnet(x1_c)

        # unpack params
        s, t, weight_logits, locs, log_scales = self._unpack_params(all_params)

        # clamp affine coefficients
        s = self.clamp * self.f_clamp(s)
        j = torch.sum(s, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            # inverse affine transformation
            y = (x1 - t) * torch.exp(-s)

            j *= -1

            # inverse logit (sigmoid) transform
            x1, ldj_inv_gauss = self._inv_gauss_inverse(x1)

            j += ldj_inv_gauss

            # inverse mixture cdf transform
            x1, ldj_mixture = gaussian_mixture_transform(
                x2, weight_logits, locs, log_scales, eps=self.inv_eps, inverse=True
            )
            ldj_mixture = sum_except_batch(ldj_mixture)
            # x1, ldj_mixture = self._mixture_cdf_inverse(
            #     x1, weight_logits, locs, log_scales
            # )

            j += ldj_mixture

        else:
            # mixture CDF transformation
            # print(locs.min(), locs.max())
            x2, ldj_mixture = gaussian_mixture_transform(
                x2, weight_logits, locs, log_scales, eps=self.inv_eps, inverse=False
            )
            # x2, ldj_mixture = self._mixture_cdf_forward(
            #     x2, weight_logits, locs, log_scales
            # )

            ldj_mixture = sum_except_batch(ldj_mixture)

            j += ldj_mixture

            # inverse Gaussian CDF (gauss prob)
            x1, ldj_inv_gauss = self._inv_gauss_forward(x2)

            j += ldj_inv_gauss

            # affine transformation
            y = torch.exp(s) * x2 + t

        return (torch.cat((x1, y), 1),), j

    def _mixture_cdf_forward(self, x, weight_logits, means, log_scale):

        z = mixture_gauss_cdf(x, weight_logits, means, log_scale)

        # mixture logistic
        log_det = mixture_gauss_log_pdf(x, weight_logits, means, log_scale)

        # log_det = sum_except_batch(log_det)

        return z, log_det

    def _mixture_cdf_inverse(self, x, weight_logits, means, log_scale):

        if self.catch_error:
            if torch.min(x) < 0 or torch.max(x) > 1:
                print(x.min(), x.max())
                raise ValueError("Values Outside Input Domain")

        # mixture cdf
        z = mixture_inv_cdf(x, weight_logits, means, log_scale, eps=self.inv_eps)

        # mixture logistic
        log_det = mixture_gauss_log_pdf(z, weight_logits, means, log_scale)

        log_det = sum_except_batch(log_det)

        return z, log_det

    def _inv_gauss_forward(self, x):

        if self.catch_error:
            if torch.min(x) < 0 or torch.max(x) > 1:
                print(x.min(), x.max())
                raise ValueError("Values Outside Input Domain")

        base_dist = dist.Normal(loc=torch.zeros(1), scale=torch.ones(1))

        # forward transformation
        z = base_dist.icdf(x)

        # log det jacobian transform
        logabsdet = -base_dist.log_prob(z)

        logabsdet = sum_except_batch(logabsdet)

        return z, logabsdet

    def _inv_gauss_inverse(self, x):

        base_dist = dist.Normal(loc=torch.zeros(1), scale=torch.ones(1))

        z = base_dist.cdf(x)

        logabsdet = -base_dist.log_prob(z)

        logabsdet = sum_except_batch(logabsdet)

        return z, logabsdet

    def output_dims(self, input_dims):
        return input_dims


def gaussian_mixture_transform(
    inputs, logit_weights, means, log_scales, eps=1e-10, max_iters=100, inverse=False
):
    """
    Univariate mixture of Gaussians transform.
    Args:
        inputs: torch.Tensor, shape (shape,)
        logit_weights: torch.Tensor, shape (shape, num_mixtures)
        means: torch.Tensor, shape (shape, num_mixtures)
        log_scales: torch.Tensor, shape (shape, num_mixtures)
        eps: float, tolerance for bisection |f(x) - z_est| < eps
        max_iters: int, maximum iterations for bisection
        inverse: bool, if True, return inverse
    """

    log_weights = F.log_softmax(logit_weights, dim=-1)
    dist = Normal(means, log_scales.exp())

    def mix_cdf(x):
        return torch.sum(log_weights.exp() * dist.cdf(x.unsqueeze(-1)), dim=-1)

    def mix_log_pdf(x):
        return torch.logsumexp(log_weights + dist.log_prob(x.unsqueeze(-1)), dim=-1)

    if inverse:
        max_scales = torch.sum(torch.exp(log_scales), dim=-1, keepdim=True)
        init_lower, _ = (means - 20 * max_scales).min(dim=-1)
        init_upper, _ = (means + 20 * max_scales).max(dim=-1)
        z = bisection_inverse(
            fn=lambda x: mix_cdf(x),
            z=inputs,
            init_x=torch.zeros_like(inputs),
            init_lower=init_lower,
            init_upper=init_upper,
            eps=eps,
            max_iters=max_iters,
        )
        ldj = mix_log_pdf(z)
        return z, ldj
    else:
        z = mix_cdf(inputs)
        ldj = mix_log_pdf(inputs)
        return z, ldj