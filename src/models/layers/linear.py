import FrEIA.modules as Fm
from nflows import transforms


class LinearLayer(Fm.InvertibleModule):
    def __init__(self, dims_in, transform: str = "householder", **kwargs):
        super().__init__(dims_in)

        assert len(dims_in[0]) <= 2
        self.transform = transform
        self.name = transform
        self.linear_transform = create_linear_transform(
            input_size=dims_in[0][0], transform=transform, **kwargs
        )

    def forward(self, x, rev=False, jac=True):
        x = x[0]
        if rev:

            z, log_det = self.linear_transform.inverse(x)
            # print(f"Mix (Out): {z.min(), z.max()}")

        else:
            # print(x.shape)
            z, log_det = self.linear_transform.forward(x)

        return (z,), log_det

    def output_dims(self, input_dims):
        return input_dims


def create_linear_transform(
    input_size: int,
    transform: str = "householder",
    num_householder: int = 10,
    identity_init: bool = True,
    using_cache: bool = True,
    with_permutation: bool = True,
    **kwargs,
) -> transforms:
    """A linear transform factory from the nflows library.
    This creates some of the standard linear transformations that one finds
    in the literature which originates from the nflows library. This transformation
    can only be used for tabular data (i.e. x = [n_samples, n_features])

    Args:
        input_size (int): the number of features for the input.
        transform (str, optional): the linear transformation (or composition) of linear transformations.
            Defaults to "householder".
        num_householder (int, optional): number of householder rotations (applies to svd and householder).
            Defaults to 10.
        identity_init (bool, optional): to initialize the transformation with the identity.
            Defaults to True.
        using_cache (bool, optional): caches the jacobian to improve the calculations.
            Defaults to True.
        with_permutation (bool, optional): optional permutation of dimensions (applies to svd and lu transformations).
            Defaults to True.


    Returns:
        transforms (nflows.transform): a nflows transformation model.

    Example:
        >>> linear_transform = create_linear_transform(
                input_size=2,
                transform="householder",
                num_householder=10
            )
    """
    if transform == "permutation":
        linear_transform = transforms.RandomPermutation(features=input_size)

    elif transform == "lu" and with_permutation:
        linear_transform = transforms.CompositeTransform(
            [
                transforms.RandomPermutation(features=input_size),
                transforms.LULinear(
                    input_size, using_cache=using_cache, identity_init=identity_init
                ),
            ]
        )
    elif transform == "lu" and not with_permutation:
        linear_transform = transforms.LULinear(
            input_size, using_cache=using_cache, identity_init=identity_init
        )

    elif transform == "svd":
        linear_transform = transforms.CompositeTransform(
            [
                transforms.RandomPermutation(features=input_size),
                transforms.SVDLinear(
                    input_size,
                    num_householder=num_householder,
                    identity_init=identity_init,
                ),
            ]
        )

    elif transform == "svd" and not with_permutation:
        linear_transform = transforms.SVDLinear(
            input_size, num_householder=num_householder, identity_init=identity_init
        )

    elif transform == "householder":

        linear_transform = transforms.HouseholderSequence(
            features=input_size, num_transforms=num_householder
        )
    else:
        raise ValueError(f"Unrecognized linear transformation {transform}")

    # initialize and return NFLows Layer
    return linear_transform
