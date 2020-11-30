from nflows import transforms


def get_marginalization(n_features: int, **kwargs):

    transforms.PiecewiseRationalQuadraticCDF(
        shape=[n_features],
        num_bins=10,
        tails="linear",
        tail_bound=10.0,
        identity_init=True,
    )
    return None