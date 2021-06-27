 
"""Logistic distribution functions."""
import torch
from torch import nn
import torch.nn.functional as F
import FrEIA.modules as Fm
from nflows.utils import sum_except_batch


class LogisticMixtureCDF(Fm.InvertibleModule):
    """Mixture-of-Logistics Coupling layer in Flow++
    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the transformation network.
        num_blocks (int): Number of residual blocks in the transformation network.
        num_components (int): Number of components in the mixture.
        drop_prob (float): Dropout probability.
        use_attn (bool): Use attention in the NN blocks.
        aux_channels (int): Number of channels in optional auxiliary input.
    """
    def __init__(self,dims_in,  n_components=4, eps=1e-5):
        super().__init__(dims_in)
        n_features = dims_in[0][0]

        self.loc = nn.Parameter(
            torch.randn(n_components, n_features), requires_grad=True
        )
        self.log_scale = nn.Parameter(
            torch.zeros(n_components, n_features), requires_grad=True
        )
        self.weight_logits = nn.Parameter(
            torch.zeros(n_components, n_features), requires_grad=True
        )

        self.n_components = n_components
        self.eps = eps
    def forward(self, x, rev=False, jac=True):
        x = x[0]

        if rev:
            raise NotImplementedError()
            # out = x_change * a.mul(-1).exp() - b
            # out, scale_ldj = logit_inverse(out, reverse=True)
            # out = out.clamp(1e-5, 1. - 1e-5)
            # out = logistic.mixture_inv_cdf(out, pi, mu, s)
            # logistic_ldj = logistic.mixture_log_pdf(out, pi, mu, s)
            # sldj = sldj - (a + scale_ldj + logistic_ldj).flatten(1).sum(-1)
        else:
            z = mixture_log_cdf(x, self.weight_logits, self.loc, self.log_scale)
            # print(z.min(), z.max())
            z=z.exp()
            # print(z.min(), z.max())
            # z, scale_ldj = logit_inverse(z)
            # out = (out + b) * a.exp()
            logistic_ldj = mixture_log_pdf(x, self.weight_logits, self.loc, self.log_scale)
            sldj = logistic_ldj #+ scale_ldj
            sldj = sum_except_batch(sldj)

        return (z,), sldj

    def output_dims(self, input_dims):
        return input_dims
    
def safe_log(x):
    return torch.log(x.clamp(min=1e-22))

def _log_pdf(x, mean, log_scale):
    """Element-wise log density of the logistic distribution."""
    z = (x - mean) * torch.exp(-log_scale)
    log_p = z - log_scale - 2 * F.softplus(z)

    return log_p


def _log_cdf(x, mean, log_scale):
    """Element-wise log CDF of the logistic distribution."""
    z = (x - mean) * torch.exp(-log_scale)
    log_p = F.logsigmoid(z)

    return log_p


def mixture_log_pdf(x, prior_logits, means, log_scales):
    """Log PDF of a mixture of logistic distributions."""
    log_ps = F.log_softmax(prior_logits, dim=1) \
        + _log_pdf(x.unsqueeze(1), means, log_scales)
    log_p = torch.logsumexp(log_ps, dim=1)

    return log_p


def mixture_log_cdf(x, prior_logits, means, log_scales):
    """Log CDF of a mixture of logistic distributions."""
    log_ps = F.log_softmax(prior_logits, dim=1) \
        + _log_cdf(x.unsqueeze(1), means, log_scales)
    log_p = torch.logsumexp(log_ps, dim=1)

    return log_p


def mixture_inv_cdf(y, prior_logits, means, log_scales,
                    eps=1e-10, max_iters=100):
    """Inverse CDF of a mixture of logisitics. Iterative algorithm."""
    # if y.min() <= 0 or y.max() >= 1:
    #     raise RuntimeError('Inverse logisitic CDF got y outside (0, 1)')
    

    def body(x_, lb_, ub_):
        cur_y = torch.exp(mixture_log_cdf(x_, prior_logits, means,
                                          log_scales))
        gt = (cur_y > y).type(y.dtype)
        lt = 1 - gt
        new_x_ = gt * (x_ + lb_) / 2. + lt * (x_ + ub_) / 2.
        new_lb = gt * lb_ + lt * x_
        new_ub = gt * x_ + lt * ub_
        return new_x_, new_lb, new_ub

    x = torch.zeros_like(y)
    lb = torch.ones_like(x) - 1_000.0
    ub = torch.ones_like(x) - 1_000.0
    # max_scales = torch.sum(torch.exp(log_scales), dim=1, keepdim=True)
    # lb, _ = (means - 20 * max_scales).min(dim=1)
    # ub, _ = (means + 20 * max_scales).max(dim=1)
    
    diff = float('inf')

    i = 0
    while diff > eps and i < max_iters:
        new_x, lb, ub = body(x, lb, ub)
        diff = (new_x - x).abs().max()
        x = new_x
        i += 1

    return x


def logit_inverse(x, reverse=False):
    """Inverse logistic function."""
    if reverse:
        z = torch.sigmoid(x)
        ldj = F.softplus(x) + F.softplus(-x)
    else:
        z = -safe_log(x.reciprocal() - 1.)
        ldj = -safe_log(x) - safe_log(1. - x)

    return z, ldj