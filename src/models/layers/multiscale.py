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
            x_split = self.prior.sample(x.shape)

            z = torch.cat([x, x_split], dim=1)

            ldj = self.prior.log_prob(x_split)

        else:
            # split inputs
            # z, z_split = self.split_input(x)
            z, z_split = torch.chunk(x, 2, dim=1)

            ldj = self.prior.log_prob(z_split)

        ldj = sum_except_batch(ldj)
        return (z,), ldj

    def output_dims(self, input_dims):
        print(input_dims)
        if len(input_dims[0]) == 1:
            d = input_dims[0]
            new_dims = d // 2
        elif len(input_dims[0]) == 3:
            c, h, w = input_dims[0]
            new_dims = (c // 2, h, w)

        else:
            raise ValueError("Errrr")
        print(new_dims)
        return [
            new_dims,
        ]
