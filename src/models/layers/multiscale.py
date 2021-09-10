"""
Code taken from:
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html#Normalizing-Flows-as-generative-model
    https://github.com/didriknielsen/survae_flows/blob/master/survae/transforms/surjections/slice.py
"""
from typing import Iterable, List
from FrEIA.modules import InvertibleModule
import torch
from nflows.utils import sum_except_batch
import numpy as np


class SplitPrior(InvertibleModule):
    """
    A simple slice layer which factors out some elements and returns
    the remaining elements for further transformation.
    This is useful for constructing multi-scale architectures [1].
    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    """

    def __init__(self, dims_in: Iterable[List[int]], prior):
        super().__init__(dims_in)
        self.prior = prior
        # self.num_keep = num_keep

    # def split_input(self, input):
    #     split_proportions = (self.num_keep, input.shape[self.dim] - self.num_keep)
    #     return torch.split(input, split_proportions, dim=self.dim)

    def forward(self, x, c=[], rev=False, jac=True):
        x = x[0]

        if rev:
            x_split = self.prior.sample(x.shape).to(x.device)

            z = torch.cat([x, x_split], dim=1)

            ldj = -self.prior.log_prob(x_split)

        else:
            # split inputs
            # z, z_split = self.split_input(x)
            z, z_split = torch.chunk(x, 2, dim=1)

            ldj = self.prior.log_prob(z_split)

        ldj = sum_except_batch(ldj)
        return (z,), ldj

    def output_dims(self, input_dims):
        if len(input_dims[0]) == 1:
            d = input_dims[0]
            new_dims = d // 2
        elif len(input_dims[0]) == 3:
            c, h, w = input_dims[0]
            new_dims = (c // 2, h, w)

        else:
            raise ValueError("Errrr")
        return [
            new_dims,
        ]


class GeneralizedSplitPrior(InvertibleModule):
    """
    A simple slice layer which factors out some elements and returns
    the remaining elements for further transformation.
    This is useful for constructing multi-scale architectures [1].
    References:
        [1] Density estimation using Real NVP,
            Dinh et al., 2017, https://arxiv.org/abs/1605.08803
    """

    def __init__(
        self, dims_in: Iterable[List[int]], prior, split: int, split_dim: int = 0
    ):
        super().__init__(dims_in)

        self.prior = prior

        if isinstance(split, int):
            # get number of dimensions in sliced dimension
            slice_dim = dims_in[0][split_dim]

            # number of dimensions to keep
            self.num_keep = split

            # number of dimensions to remove
            self.num_split = slice_dim - split

            # the dimension for the split
            self.split_dim = split_dim

        elif isinstance(split, list) or isinstance(split, tuple):

            # get number of dimensions in sliced dimension
            slice_dim = dims_in[0][split_dim]
            msg = f"splits ({split}) are not equal to total dims ({slice_dim})"
            assert slice_dim == sum(list(split)), msg

            # number of dimensions to keep
            self.num_keep = split[0]

            # number of dimensions to remove
            self.num_split = split[1]

            # the dimension for the split
            self.split_dim = split_dim
        else:
            raise ValueError(f"Unrecognized split type: {split}")

        # self.keep_dim

    def split_input(self, input):
        # split_proportions = (self.num_keep, input.shape[self.split_dim] - self.num_keep)
        return torch.split(
            input, (self.num_keep, self.num_split), dim=self.split_dim + 1
        )

    def forward(self, x, c=[], rev=False, jac=True):
        x = x[0]

        if rev:
            # get dimensions
            input_shape = list(x.shape)
            # replace input shape with correct one (include batch dim)
            input_shape[self.split_dim + 1] = self.num_split

            # sample from latent dim
            x_split = self.prior.sample(tuple(input_shape))

            z = torch.cat([x, x_split], dim=1)

            ldj = -self.prior.log_prob(x_split)

        else:
            # split inputs
            # z, z_split = self.split_input(x)
            z, z_split = self.split_input(x)

            ldj = self.prior.log_prob(z_split)

        ldj = sum_except_batch(ldj)
        return (z,), ldj

    def output_dims(self, input_dims):
        if len(input_dims[0]) == 1:
            new_dims = list(input_dims[0])
            new_dims[self.split_dim] = self.num_keep

        elif len(input_dims[0]) == 3:
            new_dims = list(input_dims[0])
            new_dims[self.split_dim] = self.num_keep

        else:
            raise ValueError("Errrr")

        return [
            tuple(new_dims),
        ]
