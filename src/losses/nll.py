import torch


def negative_log_likelihood_nats(model, x):
    return -model.log_prob(x).mean()


def negative_log_likelihood_mse(model, x):
    return -model.log_prob(x).mean()


def negative_log_likelihood_bpd(model, x):
    return -model.log_prob(x).sum() / (torch.log(2) * x.shape.numel())
