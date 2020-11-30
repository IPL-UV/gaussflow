from nflows import transforms


def get_marginalization_transform(n_features: int, squash: int = 0, **kwargs):

    if squash:
        return transforms.CompositeCDFTransform(
            transforms.Sigmoid(),
            transforms.PiecewiseRationalQuadraticCDF(shape=[n_features], **kwargs),
        )
    else:
        return transforms.PiecewiseRationalQuadraticCDF(shape=[n_features], **kwargs)


def get_rotation(
    n_features: int, rotation: str = "householder", num_householder: int = 2, **kwargs
):

    # house
    if rotation == "householder":
        return transforms.HouseholderSequence(
            features=n_features, num_transforms=num_householder
        )
    elif rotation == "pca":
        return transforms.SVDLinear(
            features=n_features,
            num_householder=num_householder,
            **kwargs,
        )

    else:
        raise ValueError(f"Unrecognized rotation: {rotation}")
