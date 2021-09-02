"""Original Code:
    https://github.com/ehoogeboom/convolution_exponential_and_sylvester/blob/main/models/transformations/conv1x1.py
"""
from typing import Iterable, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import FrEIA.modules as Fm
from src.models.layers.convolutions.conv_hh import Conv1x1Householder
from src.models.layers.convolutions.conv import Conv1x1


class ConvExponential(Fm.InvertibleModule):
    def __init__(
        self,
        dims_in: Iterable[Tuple[int]],
        n_reflections: int = 10,
        verbose: bool = False,
        n_terms: int = 6,
        householder: bool = True,
        spectral_norm: bool = False,
    ):
        super().__init__(dims_in)

        self.n_reflections = n_reflections
        self.verbose = verbose

        # extract dimensions
        self.n_channels = dims_in[0][0]
        self.H = dims_in[0][1]
        self.W = dims_in[0][2]

        # kernel weight
        kernel_size = [self.n_channels, self.n_channels, 3, 3]

        if spectral_norm:
            raise NotImplementedError
        else:
            self.kernel = torch.nn.Parameter(
                torch.randn(kernel_size) / np.prod(kernel_size[1:])
            )

        self.stride = (1, 1)
        self.padding = (1, 1)

        # add householder convolution
        if householder:
            self.conv1x1 = Conv1x1Householder(
                dims_in=dims_in, n_reflections=n_reflections
            )
        else:
            self.conv1x1 = Conv1x1(dims_in=dims_in)

        # initialize matrix
        self.n_terms_train = n_terms
        self.n_terms_eval = self.n_terms_train * 2 + 1

    def forward(self, x, rev=False, jac=True):
        x = x[0]

        n_samples, *_ = x.size()

        kernel = self.kernel

        #
        n_terms = self.n_terms_train if self.training else self.n_terms_eval

        if not rev:
            # 1x1 convolution + householder

            z, ldj = self.conv1x1([x], rev=False)

            z = z[0]

            # convolution exponential
            z = conv_exp(z, kernel, terms=n_terms, verbose=self.verbose)

            # add log det jacobian term
            ldj = ldj + log_det(kernel) * self.H * self.W

        else:
            if x.device != kernel.device:
                print("Warning, x.device is not kernel.device")
                kernel = kernel.to(device=x.device)

            # inverse convolutional exponential
            z = inv_conv_exp(x, kernel, terms=n_terms, verbose=self.verbose)

            ldj = log_det(kernel) * self.H * self.W

            # inverse convolutional layer
            z, ldj_conv = self.conv1x1([z], rev=True)
            z = z[0]

            ldj = ldj_conv + log_det(kernel) * self.H * self.W

            # calculate log det jacobian

        return (z,), ldj

    def inverse(self, x, rev=True, jac=True):
        return self(x, rev=True, jac=jac)

    def output_dims(self, input_dims):
        return input_dims


def matrix_log(B, terms=10):
    assert B.size(0) == B.size(1)
    I = torch.eye(B.size(0))

    B_min_I = B - I

    # for k = 1.
    product = B_min_I
    result = B_min_I

    is_minus = -1
    for k in range(2, terms):
        # Reweighing with k term.
        product = torch.matmul(product, B_min_I) * (k - 1) / k
        result = result + is_minus * product

        is_minus *= -1

    return result


def matrix_exp(M, terms=10):
    assert M.size(0) == M.size(1)
    I = torch.eye(M.size(0))

    # for i = 0.
    result = I
    product = I

    for i in range(1, terms + 1):
        product = torch.matmul(product, M) / i
        result = result + product

    return result


def conv_exp(input, kernel, terms=10, dynamic_truncation=0, verbose=False):
    B, C, H, W = input.size()

    assert kernel.size(0) == kernel.size(1)
    assert kernel.size(0) == C, "{} != {}".format(kernel.size(0), C)

    padding = (kernel.size(2) - 1) // 2, (kernel.size(3) - 1) // 2

    result = input
    product = input

    for i in range(1, terms + 1):
        product = F.conv2d(product, kernel, padding=padding, stride=(1, 1)) / i
        result = result + product

        if dynamic_truncation != 0 and i > 5:
            if product.abs().max().item() < dynamic_truncation:
                break

    if verbose:
        print("Maximum element size in term: {}".format(torch.max(torch.abs(product))))

    return result


def inv_conv_exp(input, kernel, terms=10, dynamic_truncation=0, verbose=False):
    return conv_exp(input, -kernel, terms, dynamic_truncation, verbose)


def log_det(kernel):
    Cout, C, K1, K2 = kernel.size()
    assert Cout == C

    M1 = (K1 - 1) // 2
    M2 = (K2 - 1) // 2

    diagonal = kernel[torch.arange(C), torch.arange(C), M1, M2]

    trace = torch.sum(diagonal)

    return trace


def convergence_scale(c, kernel_size):
    C_out, C_in, K1, K2 = kernel_size

    d = C_in * K1 * K2

    return c / d
