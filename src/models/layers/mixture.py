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

class GaussianMixtureCDF(Fm.InvertibleModule):
    def __init__(self,dims_in,  n_components=4, eps=1e-5, init_X=None, inv_max_iters: int=100, inv_eps: float=1e-10, catch_error: bool=False):
        super().__init__(dims_in)
        n_features = dims_in[0][0]

        if init_X is not None:
            prior_logits, means, log_scales = init_mixture_weights(init_X, n_features=init_X.shape[1], n_components=n_components)
            self.loc = nn.Parameter(
                torch.Tensor(means), requires_grad=True
            )
            self.log_scale = nn.Parameter(
                torch.Tensor(log_scales), requires_grad=True
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
        self.catch_error = catch_error

    def forward(self, x, rev=False, jac=True):
        x = x[0]
        if rev:

            if self.catch_error:
                if torch.min(x) < 0 or torch.max(x) > 1:
                    print(x.min(), x.max())
                    raise InputOutsideDomain()

            x = torch.clamp(x, self.eps, 1 - self.eps)

            z = mixture_inv_cdf(x, self.weight_logits, self.loc, self.log_scale, eps=self.inv_eps)
            log_det = mixture_log_pdf(z,  self.weight_logits, self.loc, self.log_scale)
            # print(f"Mix (Out): {z.min(), z.max()}")

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
    component_dist = dist.Normal(loc=means, scale=log_scales.exp())

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
    component_dist = dist.Normal(loc=means, scale=log_scales.exp())

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

def init_mixture_weights(X, n_features, n_components,**kwargs):


    prior_logits, means, covariances = init_means_GMM_marginal(
        X,
        n_components=n_components,
        covariance_type="diag",
        reg_covar=1e-5,
        **kwargs,
    )
    log_scales = softplus_inverse(np.sqrt(covariances))

    prior_logits = np.array(prior_logits)
    prior_logits = np.log(prior_logits)

    means = np.array(means)

    return prior_logits, means, log_scales


def init_means_GMM_marginal(X: np.ndarray, n_components: int, **kwargs):
    """Initialize means with K-Means
    
    Parameters
    ----------
    X : np.ndarray
        (n_samples, n_features)
    n_components : int
        the number of clusters for the K-Means
    
    Returns
    -------
    clusters : np.ndarray
        (n_features, n_components)"""

    weights, means, covariances = [], [], []

    for iX in X.T:
        clf = GaussianMixture(n_components=n_components, **kwargs).fit(iX[:, None])
        weights.append(clf.weights_)
        means.append(clf.means_.T)
        covariances.append(clf.covariances_.T)

    return np.vstack(weights), np.vstack(means), np.vstack(covariances)


def softplus_inverse(x):
    return np.log(np.exp(x) - 1.0)


