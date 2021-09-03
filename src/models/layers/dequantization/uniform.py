"""
Code taken from:
    https://github.com/didriknielsen/survae_flows/blob/master/survae/transforms/surjections/dequantization_uniform.py
"""
from typing import Iterable, List
from FrEIA.modules import InvertibleModule
import torch
from nflows.utils import sum_except_batch


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
