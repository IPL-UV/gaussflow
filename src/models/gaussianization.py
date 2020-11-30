from nflows import transforms


def get_marginalization_transform(n_features: int, squash: int = 0, **kwargs):

    if squash:
        transforms.CompositeCDFTransform(
            transforms.Sigmoid(),
            transforms.PiecewiseRationalQuadraticCDF(
                shape=[1],
                tail_bound=1.0,
            ),
        )
    else:
        mg_gauss = transforms.PiecewiseRationalQuadraticCDF(
            shape=[n_features], tails="linear", tail_bound=10.0, identity_init=True, **kwargs
        )
    return mg_gauss
