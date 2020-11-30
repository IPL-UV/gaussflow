from nflows import transforms


def get_marginalization_transform(n_features: int, squash: int = 0, **kwargs):

    if squash:
        mg_gauss = transforms.CompositeCDFTransform(
            transforms.Sigmoid(),
            transforms.PiecewiseRationalQuadraticCDF(
                shape=[n_features], tail_bound=1.0, **kwargs
            ),
        )
    else:
        mg_gauss = transforms.PiecewiseRationalQuadraticCDF(
            shape=[n_features], **kwargs
        )
    return mg_gauss
