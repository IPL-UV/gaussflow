import torch
from torch import nn
import FrEIA.modules as Fm
import torch.nn.functional as F
from torch.distributions.normal import Normal
import torch.distributions as dist
from nflows.utils import sum_except_batch
from einops import repeat
import numpy as np
from sklearn.mixture import GaussianMixture
from src.models.layers.mixture import init_mixture_weights

class LogisticMixtureCDF(Fm.InvertibleModule):
    def __init__(self,dims_in,  n_components=4, eps=1e-5, init_X=None, inv_max_iters: int=100, inv_eps: float=1e-10, clamp_inverse: bool=True):
        super().__init__(dims_in)
        n_features = dims_in[0][0]

        if init_X is not None:
            prior_logits, means, log_scales = init_mixture_weights(init_X, n_features=init_X.shape[1], n_components=n_components)
            self.loc = nn.Parameter(
                torch.Tensor(means), requires_grad=True
            )
            self.log_scale = nn.Parameter(
                torch.Tensor(0.5 * log_scales), requires_grad=True
            )
            self.weight_logits = nn.Parameter(
                torch.Tensor(prior_logits), requires_grad=True
            )

        else:
            self.loc = nn.Parameter(
                torch.randn(n_features, n_components), requires_grad=True
            )
            self.log_scale = nn.Parameter(
                torch.zeros(n_features, n_components), requires_grad=True
            )
            self.weight_logits = nn.Parameter(
                torch.zeros(n_features, n_components), requires_grad=True
            )

        self.n_components = n_components
        self.eps = eps
        self.inv_max_iters = inv_max_iters
        self.inv_eps = inv_eps
        self.clamp_inverse = clamp_inverse

    def forward(self, x, rev=False, jac=True):
        x = x[0]
        if rev:

            if self.clamp_inverse:
                x = torch.clamp(x, self.eps, 1 - self.eps)

            z = mixture_inv_cdf(x, self.weight_logits, self.loc, self.log_scale)
            log_det = mixture_log_pdf(z,  self.weight_logits, self.loc, self.log_scale)
            # print(f"Mix (Out): {z.min(), z.max()}")
            log_det = sum_except_batch(log_det)
            print(z.min(), z.max())

        else:
            # print(x.shape)
            x = repeat(x, "N D -> N D K", K=self.n_components)
            # print(f"x - [N,D,K]:", x.shape)

            # Initialize the mixture distribution with the mean/loc and std/scale parameters.
            mix_dist = dist.Categorical(self.weight_logits)
            component_dist = dist.Normal(loc=self.loc, scale=self.log_scale.exp())
            # print(f"mu - [N,D,K]:", self.loc.shape)
            # print(f"sigma - [N,D,K]:", self.log_scale.shape)

            # CDF Distribution
            mix_prob = mix_dist.probs
            z_cdf = component_dist.cdf(x)
            # print(z_cdf.shape, mix_prob.shape)
            z = torch.sum(z_cdf * mix_prob, dim=-1)

            # Log Probability
            z_log_prob = component_dist.log_prob(x)
            log_mix_prob = torch.log_softmax(mix_dist.logits, dim=-1) 
            log_det = torch.logsumexp(z_log_prob + log_mix_prob, dim=-1) 
            log_det = sum_except_batch(log_det)

        return (z,), log_det

    def output_dims(self, input_dims):
        return input_dims

def mixture_log_pdf(x, weights, means, log_scales):
    # print(x.shape)
    x = repeat(x, "N D -> N D K", K=weights.shape[1])
    # print(f"x - [N,D,K]:", x.shape)

    # Initialize the mixture distribution with the mean/loc and std/scale parameters.
    mix_dist = dist.Categorical(weights)
    base_dist = dist.Uniform(0, 1)
    transforms = [dist.SigmoidTransform().inv, dist.AffineTransform(loc=means, scale=log_scales.exp())]
    component_dist = dist.TransformedDistribution(base_dist, transforms)

    # CDF Distribution
    log_prob = component_dist.log_prob(x)
    log_mix_prob = torch.log_softmax(mix_dist.logits, dim=-1) 
    log_det = torch.logsumexp(log_prob + log_mix_prob, dim=-1) 

    return log_det

def mixture_log_cdf(x, weights, means, log_scales):
    # print(x.shape)
    x = repeat(x, "N D -> N D K", K=weights.shape[1])
    # print(f"x - [N,D,K]:", x.shape)

    # Initialize the mixture distribution with the mean/loc and std/scale parameters.
    mix_dist = dist.Categorical(weights)
    base_dist = dist.Uniform(0, 1)
    transforms = [dist.SigmoidTransform().inv, dist.AffineTransform(loc=means, scale=log_scales.exp())]
    component_dist = dist.TransformedDistribution(base_dist, transforms)


    # CDF Distribution
    mix_prob = mix_dist.probs
    z_cdf = component_dist.cdf(x)
    # print(z_cdf.shape, mix_prob.shape)
    z = torch.sum(z_cdf * mix_prob, dim=-1)

    return z

def mixture_inv_cdf(x, prior_logits, means, log_scales,
                    eps=1e-10, max_iters=100):
    """Inverse CDF of a mixture of logisitics. Iterative algorithm."""
    # if y.min() <= 0 or y.max() >= 1:
    #     raise RuntimeError('Inverse logisitic CDF got y outside (0, 1)')
    

    """Inverse CDF of a mixture of logisitics. Iterative algorithm."""
    if x.min() <= 0 or x.max() >= 1:
        # print(x.min(), x.max())
        raise RuntimeError('Inverse logisitic CDF got y outside (0, 1)')

    def body(x_, lb_, ub_):
        cur_y = mixture_log_cdf(x_, prior_logits, means, log_scales)

        gt = (cur_y > x).type(x.dtype)
        lt = 1 - gt
        new_x_ = gt * (x_ + lb_) / 2. + lt * (x_ + ub_) / 2.
        new_lb = gt * lb_ + lt * x_
        new_ub = gt * x_ + lt * ub_
        return new_x_, new_lb, new_ub

    z = torch.zeros_like(x)
    max_scales = torch.sum(torch.exp(log_scales), dim=1, keepdim=True)
    lb, _ = (means - 20 * max_scales).min(dim=1)
    lb = lb * torch.ones_like(x)
    ub, _ = (means + 20 * max_scales).max(dim=1)
    ub = ub * torch.ones_like(x)
    # x = torch.zeros_like(y)
    # lb = torch.ones_like(x) - 1_000.0
    # ub = torch.ones_like(x) - 1_000.0
    diff = float('inf')
    # print(x.shape, lb.shape)

    i = 0
    while diff > eps and i < max_iters:
        new_z, lb, ub = body(z, lb, ub)
        diff = (new_z - z).abs().max()
        z = new_z
        i += 1

    return z
