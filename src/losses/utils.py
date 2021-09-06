import torch
import numpy as np


def stable_avg_log_probs(data):

    n_samples = data.size()[-1]

    return torch.logsumexp(data, dim=-1) - np.log(n_samples)
