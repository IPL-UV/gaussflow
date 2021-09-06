import numpy as np
from sklearn.utils import check_random_state


def gen_bimodal_data(n_samples=512):
    """Toy Univariate Bimodal Dataset"""
    return np.r_[
        np.random.randn(n_samples // 2, 1) + np.array([2]),
        np.random.randn(n_samples // 2, 1) + np.array([-2]),
    ]


def get_bivariate_data(
    dataset: str = "rbig", n_samples: int = 2_000, noise: float = 0.05, seed: int = 123
):
    if dataset == "rbig":
        return rbig_data(n_samples=n_samples, noise=noise, seed=seed)
    else:
        raise ValueError(f"Unrecognized dataset: {dataset}")


def rbig_data(n_samples: int, noise: float = 0.05, seed: int = 123):
    rng = check_random_state(seed)
    X = np.abs(2 * rng.randn(n_samples, 1))
    Y = np.sin(X) + noise * rng.randn(n_samples, 1)
    data = np.hstack((X, Y))
    return data