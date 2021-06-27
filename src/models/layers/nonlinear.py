import torch
from torch import nn
import torch.nn.functional as F
import FrEIA.modules as Fm
import torch.distributions as dist
import math
from nflows.utils import sum_except_batch
from nflows.transforms.base import InputOutsideDomain

class InverseGaussCDF(Fm.InvertibleModule):
    def __init__(self, dim_in, eps=1e-6, catch_error=False):
        super().__init__(dim_in)
        self.eps = eps
        self.base_dist = dist.Normal(loc=torch.zeros(1), scale=torch.ones(1))
        self.catch_error = catch_error
    def forward(self, x, rev=False, jac=True):

        x = x[0]

        if rev:
            # print(f"ICDF (In): {x.min(), x.max()}")
            z = self.base_dist.cdf(x)
            logabsdet = -self.base_dist.log_prob(z)
            # print(f"ICDF: {z.min(), z.max()}")

            return (z,), logabsdet
        else:
            # print(x.min(), x.max())/
            if self.catch_error:
                if torch.min(x) < 0 or torch.max(x) > 1:
                    print(x.min(), x.max())
                    raise InputOutsideDomain()

            x = torch.clamp(x, self.eps, 1 - self.eps)

            z = self.base_dist.icdf(x)

            # logabsdet = - _stable_log_pdf(z)
            logabsdet = -self.base_dist.log_prob(z)
            logabsdet = sum_except_batch(logabsdet)

            return (z,), logabsdet


    def output_dims(self, input_dims):
        return input_dims

class Sigmoid(Fm.InvertibleModule):
    def __init__(self, dim_in, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__(dim_in)
        self.eps = eps
        if learn_temperature:
            self.temperature = nn.Parameter(torch.Tensor([temperature]))
        else:
            self.temperature = torch.Tensor([temperature])

    def forward(self, x, rev=False, jac=True):

        if not rev:
            inputs = self.temperature * inputs
            outputs = torch.sigmoid(inputs)
            logabsdet = sum_except_batch(
                torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs)
            )
            return (outputs,), logabsdet
        else:
            # if torch.min(inputs) < 0 or torch.max(inputs) > 1:
            #     raise InputOutsideDomain()

            inputs = torch.clamp(inputs, self.eps, 1 - self.eps)

            outputs = (1 / self.temperature) * (torch.log(inputs) - torch.log1p(-inputs))
            logabsdet = -sum_except_batch(
                torch.log(self.temperature)
                - F.softplus(-self.temperature * outputs)
                - F.softplus(self.temperature * outputs)
            )
            return (outputs,), logabsdet


    def output_dims(self, input_dims):
        return input_dims

class Logit(Fm.InvertibleModule):
    def __init__(self, dim_in, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__(dim_in)
        self.eps = eps
        if learn_temperature:
            self.temperature = nn.Parameter(torch.Tensor([temperature]))
        else:
            self.temperature = torch.Tensor([temperature])

    def forward(self, x, rev=False, jac=True):

        inputs = x[0]

        if not rev:
            inputs = torch.clamp(inputs, self.eps, 1 - self.eps)

            outputs = (1 / self.temperature) * (torch.log(inputs) - torch.log1p(-inputs))
            logabsdet = - sum_except_batch(
                torch.log(self.temperature)
                - F.softplus(-self.temperature * outputs)
                - F.softplus(self.temperature * outputs)
            )

        else:
            # print(x.min(), x.max())
            # if torch.min(x) < 0 or torch.max(x) > 1:
            #     raise InputOutsideDomain()

            inputs = self.temperature * inputs
            outputs = torch.sigmoid(inputs)
            logabsdet = sum_except_batch(
                torch.log(self.temperature) - F.softplus(-inputs) - F.softplus(inputs)
            )
        return (outputs,), logabsdet


    def output_dims(self, input_dims):
        return input_dims

def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    # if not check.is_nonnegative_int(num_batch_dims):
    #     raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)

_half_log2pi = 0.5 * math.log(2 * math.pi)

def _stable_log_pdf(x):

    log_unnormalized = -0.5 * x.pow(2)

    log_normalization = _half_log2pi
    return log_unnormalized - log_normalization