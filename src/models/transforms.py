import torch
from torch import nn
from scipy.optimize import bisect
import torch.nn.functional as F
from torch.distributions.normal import Normal
from nflows.transforms.base import Transform


class GaussianMixtureCDF(Transform):
    def __init__(self, n_features, n_components=4, eps=1e-5):
        super().__init__()

        self.loc = nn.Parameter(
            torch.randn(n_components, n_features), requires_grad=True
        )
        self.log_scale = nn.Parameter(
            1.5 * torch.tanh(torch.zeros(n_components, n_features)), requires_grad=True
        )
        self.weight_logits = nn.Parameter(
            torch.zeros(n_components, n_features), requires_grad=True
        )

        self.n_components = n_components
        self.eps = eps

    def forward(self, x, context=None):
        # set up mixture distribution
        # print(self.weight_logits.shape)
        weights = F.softmax(self.weight_logits, dim=0)
        # print(f"Weights: {weights.shape}")
        weights = weights.unsqueeze(0)
        # print(f"Weights: {weights.shape}")
        # weights = weights.repeat(x.shape[0], 1)
        # print(f"Weights: {weights.shape}")
        mixture_dist = Normal(self.loc, self.log_scale.exp())
        # print(mixture_dist.shape)
        x = x.unsqueeze(1)  # .repeat(1, self.n_components)
        # print(f"x: {x.shape}")

        # z = cdf of x
        z = mixture_dist.cdf(x) * weights
        # print(f"logcdfs: {z.shape, z.min(), z.max()}")
        z = torch.sum(z, dim=1)
        # print(f"logcdfs: {z.shape, z.min(), z.max()}")
        z = torch.clamp(z, self.eps, 1 - self.eps)

        # print(f"Z: {z.shape}")
        # log_det = log dz/dx = log pdf(x)
        log_det = mixture_dist.log_prob(x).exp() * weights
        log_det = torch.sum(log_det, dim=1).log()
        log_det = log_det.sum(dim=-1)

        return z, log_det

    def inverse(self, inputs, context=None):
        # Find the exact x via bisection such that f(x) = z
        results = []
        for z_elem in inputs:

            def f(x):
                return self.forward(torch.tensor(x).unsqueeze(0))[0] - z_elem

            x = bisect(f, -20, 20)
            results.append(x)
        return (
            torch.tensor(results).reshape(inputs.shape),
            torch.tensor(results).reshape(inputs.shape[0]),
        )
