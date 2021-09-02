from tqdm import trange

# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch

from src.models.layers.rqs import RQSplines
from src.models.layers.mixtures import GaussianMixtureCDF
from src.models.layers.mixtures import LogisticMixtureCDF
from src.models.layers.nonlinear import InverseGaussCDF, Logit


def init_gaussianization_flow(
    X_init: torch.Tensor,
    non_linear: str = "gaussian",
    n_components: int = 10,
    n_layers: int = 10,
    n_reflections: int = 2,
):

    init_X = [
        [X_init],
    ]

    n_features = X_init.shape[1]

    inn = Ff.SequenceINN(n_features)

    for _ in trange(n_layers):

        # init Gaussian Mixture CDF
        if non_linear == "gaussian":
            # Gaussian Mixture Layer
            inn.append(
                GaussianMixtureCDF,
                n_components=n_components,
                init_X=init_X[0][0],
            )
            # forward transformation
            with torch.no_grad():
                init_X = inn.module_list[-1](x=init_X[0])

            # Inverse CDF Transform
            inn.append(InverseGaussCDF)
        elif non_linear == "logistic":
            # Gaussian Mixture Layer
            inn.append(
                LogisticMixtureCDF,
                n_components=n_components,
                init_X=init_X[0][0],
            )
            # forward transformation
            with torch.no_grad():
                init_X = inn.module_list[-1](x=init_X[0])

            # Inverse CDF Transform
            inn.append(Logit)
        elif non_linear == "splines":
            # Gaussian Mixture Layer
            inn.append(RQSplines, num_bins=n_components, tail_bound=12, tails="linear")
        else:
            raise ValueError(f"Unrecognized model: {non_linear}")
        with torch.no_grad():
            init_X = inn.module_list[-1](x=init_X[0])

        # Householder Transformation
        inn.append(Fm.HouseholderPerm, n_reflections=n_reflections)
        with torch.no_grad():
            init_X = inn.module_list[-1](x=init_X[0])

    return inn
