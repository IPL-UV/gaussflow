"""
Code taken from:
    https://github.com/didriknielsen/survae_flows/blob/master/survae/transforms/surjections/dequantization_uniform.py
"""
from typing import Iterable, List
from FrEIA.modules import InvertibleModule
import torch
from nflows.utils import sum_except_batch
import numpy as np
import torch.nn.functional as F


class UniformDequantization(InvertibleModule):
    """
    A uniform dequantization layer.
    This is useful for converting discrete variables to continuous [1, 2].
    Forward:
        `z = (x+u)/K, u~Unif(0,1)^D`
        where `x` is discrete, `x \in {0,1,2,...,K-1}^D`.
    Inverse:
        `x = Quantize(z, K)`
    Args:
        num_bits: int, number of bits in quantization,
            i.e. 8 for `x \in {0,1,2,...,255}^D`
            or 5 for `x \in {0,1,2,...,31}^D`.
    References:
        [1] RNADE: The real-valued neural autoregressive density-estimator,
            Uria et al., 2013, https://arxiv.org/abs/1306.0186
        [2] Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design,
            Ho et al., 2019, https://arxiv.org/abs/1902.00275
    """

    stochastic_forward = True

    def __init__(self, dims_in: Iterable[List[int]], num_bits: int = 8):
        super().__init__(dims_in)
        self.num_bits = num_bits
        self.quantization_bins = 2 ** num_bits
        self.register_buffer(
            "ldj_per_dim",
            -torch.log(torch.tensor(self.quantization_bins, dtype=torch.float)),
        )

    def _ldj(self, shape):
        batch_size = shape[0]
        num_dims = shape[1:].numel()
        ldj = self.ldj_per_dim * num_dims
        return ldj.repeat(batch_size)

    def forward(self, x, rev=False, jac=True):
        x = x[0]

        if rev:
            z = self.quantization_bins * x

            z = z.floor().clamp(min=0, max=self.quantization_bins - 1).long()

        else:
            u = torch.rand(
                x.shape, device=self.ldj_per_dim.device, dtype=self.ldj_per_dim.dtype
            )
            z = (x.type(u.dtype) + u) / self.quantization_bins
        ldj = self._ldj(z.shape)

        ldj = sum_except_batch(ldj)
        return (z,), ldj

    def output_dims(self, input_dims):
        return input_dims


# class UniformDequantization(InvertibleModule):
#     """
#     A uniform dequantization layer.
#     This is useful for converting discrete variables to continuous [1, 2].
#     Forward:
#         `z = (x+u)/K, u~Unif(0,1)^D`
#         where `x` is discrete, `x \in {0,1,2,...,K-1}^D`.
#     Inverse:
#         `x = Quantize(z, K)`
#     Args:
#         num_bits: int, number of bits in quantization,
#             i.e. 8 for `x \in {0,1,2,...,255}^D`
#             or 5 for `x \in {0,1,2,...,31}^D`.
#     References:
#         [1] RNADE: The real-valued neural autoregressive density-estimator,
#             Uria et al., 2013, https://arxiv.org/abs/1306.0186
#         [2] Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design,
#             Ho et al., 2019, https://arxiv.org/abs/1902.00275
#     """

#     stochastic_forward = True

#     def __init__(
#         self, dims_in: Iterable[List[int]], num_bits: int = 8, alpha: float = 1e-5
#     ):
#         super().__init__(dims_in)
#         self.num_bits = num_bits
#         self.quantization_bins = 2 ** num_bits
#         self.alpha = alpha
#         # self.register_buffer(
#         #     "ldj_per_dim",
#         #     -torch.log(torch.tensor(self.quantization_bins, dtype=torch.float)),
#         # )

#     def forward(self, x, rev=False, jac=True):

#         x = x[0]
#         if not rev:
#             z, ldj = self.dequant(x)
#             z, ldj_sigmoid = self.sigmoid(z, reverse=True)
#             ldj += ldj_sigmoid
#         else:
#             z, ldj = self.sigmoid(x, reverse=False)
#             z = z * self.quantization_bins
#             ldj += np.log(self.quantization_bins) * np.prod(z.shape[1:])
#             z = (
#                 torch.floor(z)
#                 .clamp(min=0, max=self.quantization_bins - 1)
#                 .to(torch.int32)
#             )
#         ldj = sum_except_batch(ldj)
#         return (z,), ldj

#     def sigmoid(self, z, reverse=False):
#         # Applies an invertible sigmoid transformation
#         if not reverse:
#             ldj = -z - 2 * F.softplus(-z)
#             z = torch.sigmoid(z)
#         else:
#             z = (
#                 z * (1 - self.alpha) + 0.5 * self.alpha
#             )  # Scale to prevent boundaries 0 and 1
#             ldj = np.log(1 - self.alpha) * np.prod(z.shape[1:])
#             ldj += -torch.log(z) - torch.log(1 - z)
#             z = torch.log(z) - torch.log(1 - z)

#         ldj = sum_except_batch(ldj)
#         return z, ldj

#     def dequant(self, z):
#         # Transform discrete values to continuous volumes
#         z = z.to(torch.float32)
#         z = z + torch.rand_like(z).detach()
#         z = z / self.quantization_bins
#         ldj = np.log(self.quantization_bins) * np.prod(z.shape[1:])
#         return z, -ldj

#     def output_dims(self, input_dims):
#         return input_dims
