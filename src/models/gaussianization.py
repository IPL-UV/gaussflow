from tqdm import trange

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch

from src.models.layers.splines import RationalQuadraticSplines, RationalLinearSplines
from src.models.layers.mixtures import GaussianMixtureCDF
from src.models.layers.mixtures import LogisticMixtureCDF
from src.models.layers.nonlinear import InverseGaussCDF, Logit
from src.experiments.utils import gf_propagate
from src.models.layers.convolutions import Conv1x1Householder


def init_gaussianization_flow(
    inn,
    init_X: torch.Tensor,
    non_linear: str = "gaussian",
    n_components: int = 10,
    n_reflections: int = 2,
    tail_bound: int = 12,
):

    n_features = init_X.shape[1:]

    # init Gaussian Mixture CDF
    if non_linear == "gaussian":
        # Gaussian Mixture Layer
        inn.append(
            GaussianMixtureCDF,
            n_components=n_components,
            init_X=init_X,
        )
        # forward transformation
        init_X = gf_propagate(inn, init_X)

        # Inverse CDF Transform
        inn.append(InverseGaussCDF)

    elif non_linear == "logistic":
        # Gaussian Mixture Layer
        inn.append(
            LogisticMixtureCDF,
            n_components=n_components,
            init_X=init_X,
        )
        # forward transformation
        init_X = gf_propagate(inn, init_X)

        # Inverse CDF Transform
        inn.append(Logit)

    elif non_linear == "rqsplines":
        # Gaussian Mixture Layer
        inn.append(
            RationalQuadraticSplines,
            num_bins=n_components,
            tail_bound=tail_bound,
            tails="linear",
        )
    elif non_linear == "rlsplines":
        # Gaussian Mixture Layer
        inn.append(
            RationalLinearSplines,
            num_bins=n_components,
            tail_bound=tail_bound,
            tails="linear",
        )
    else:
        raise ValueError(f"Unrecognized model: {non_linear}")

    with torch.no_grad():
        init_X = gf_propagate(inn, init_X)

    # Householder Transformation
    inn.append(
        Fm.HouseholderPerm,
        n_reflections=n_reflections,
    )

    init_X = gf_propagate(inn, init_X)

    return inn, init_X


def init_gaussianization_image_flow(
    inn,
    init_X: torch.Tensor,
    non_linear: str = "gaussian",
    n_components: int = 10,
    n_reflections: int = 2,
    tail_bound: int = 12,
):

    n_channels, height, width = init_X.shape[1:]

    inn.append(Fm.Flatten)
    init_X = gf_propagate(inn, init_X)

    # init Gaussian Mixture CDF
    if non_linear == "gaussian":
        # Gaussian Mixture Layer
        inn.append(
            GaussianMixtureCDF,
            n_components=n_components,
            init_X=init_X,
        )
        # forward transformation
        init_X = gf_propagate(inn, init_X)

        # Inverse CDF Transform
        inn.append(InverseGaussCDF)

    elif non_linear == "logistic":
        # Gaussian Mixture Layer
        inn.append(
            LogisticMixtureCDF,
            n_components=n_components,
            init_X=init_X,
        )
        # forward transformation
        init_X = gf_propagate(inn, init_X)

        # Inverse CDF Transform
        inn.append(Logit)

    elif non_linear == "rqsplines":
        # Gaussian Mixture Layer
        inn.append(
            RationalQuadraticSplines,
            num_bins=n_components,
            tail_bound=tail_bound,
            tails="linear",
        )
    elif non_linear == "rlsplines":
        # Gaussian Mixture Layer
        inn.append(
            RationalLinearSplines,
            num_bins=n_components,
            tail_bound=tail_bound,
            tails="linear",
        )
    else:
        raise ValueError(f"Unrecognized model: {non_linear}")

    init_X = gf_propagate(inn, init_X)

    # UNFLATTEN
    inn.append(Fm.Reshape, output_dims=(n_channels, height, width))

    init_X = gf_propagate(inn, init_X)

    # Householder Transformation
    inn.append(
        Conv1x1Householder,
        n_reflections=n_reflections,
    )

    init_X = gf_propagate(inn, init_X)

    return inn, init_X
