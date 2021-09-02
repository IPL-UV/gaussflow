import torch
import FrEIA.modules as Fm
import torch.distributions as dist
from nflows.utils import sum_except_batch
from src.models.layers.utils import InputOutsideDomain
from nflows.transforms.nonlinearities import (
    Logit as NFlowsLogit,
)
from src.models.layers.utils import NFlowsLayer


class InverseGaussCDF(Fm.InvertibleModule):
    def __init__(self, dim_in, eps=1e-6, catch_error=False):
        super().__init__(dim_in)
        self.eps = eps
        self.base_dist = dist.Normal(loc=torch.zeros(1), scale=torch.ones(1))
        self.catch_error = catch_error

    def forward(self, x, rev=False, jac=True):

        x = x[0]

        if rev:

            z = self.base_dist.cdf(x)

            logabsdet = -self.base_dist.log_prob(z)

        else:

            if self.catch_error:
                if torch.min(x) < 0 or torch.max(x) > 1:
                    print(x.min(), x.max())
                    raise InputOutsideDomain()

            x = torch.clamp(x, self.eps, 1 - self.eps)

            z = self.base_dist.icdf(x)

            logabsdet = -self.base_dist.log_prob(z)

        logabsdet = sum_except_batch(logabsdet)

        return (z,), logabsdet

    def output_dims(self, input_dims):
        return input_dims


class Logit(NFlowsLayer):
    def __init__(self, dim_in, temperature=1.0, eps=1e-5):

        transform = NFlowsLogit(temperature=temperature, eps=eps)

        super().__init__(dim_in, transform=transform, name="logit")


# _half_log2pi = 0.5 * math.log(2 * math.pi)


# def _stable_log_pdf(x):

#     log_unnormalized = -0.5 * x.pow(2)

#     log_normalization = _half_log2pi
#     return log_unnormalized - log_normalization