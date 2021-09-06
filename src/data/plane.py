from typing import Optional, Tuple

import numpy as np
from sklearn import datasets
from torch.utils.data import DataLoader, Dataset
import torch

# import torch.multiprocessing as multiprocessing

# multiprocessing.set_start_method("spawn")
# class DensityDataset:
#     def __init__(self, data, dtype=np.float32):
#         self.data = data
#         self.dtype = dtype

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx: int) -> np.ndarray:
#         data = self.data[idx]
#         return np.np.ndarray(data, dtype=self.dtype)


def add_dataset_args(parser):
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="number of data points for training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="noisysine",
        help="Dataset to be used for visualization",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5_000,
        help="number of data points for training",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="number of data points for training",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=2_000,
        help="number of data points for training",
    )
    parser.add_argument(
        "--n_valid",
        type=int,
        default=1_000,
        help="number of data points for training",
    )
    return parser


class DensityDataset(Dataset):
    def __init__(self, n_samples: int = 10_000, noise: float = 0.1, seed: int = 123):
        self.n_samples = n_samples
        self.seed = seed
        self.noise = noise
        self.reset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = self.data[idx]
        return data

    def reset(self):
        self._create_data()

    def _create_data(self):
        raise NotImplementedError


class GenericDataset(DensityDataset):
    def __init__(self, data):
        self.data = data


def get_data(
    N: int = 30,
    input_noise: float = 0.15,
    output_noise: float = 0.15,
    N_test: int = 400,
) -> Tuple[np.ndnp.ndarray, np.ndnp.ndarray, np.ndnp.ndarray]:
    np.random.seed(0)
    X = np.linspace(-1, 1, N)
    Y = X + 0.2 * np.power(X, 3.0) + 0.5 * np.power(0.5 + X, 2.0) * np.sin(4.0 * X)
    Y += output_noise * np.random.randn(N)
    Y -= np.mean(Y)
    Y /= np.std(Y)

    X += input_noise * np.random.randn(N)

    assert X.shape == (N,)
    assert Y.shape == (N,)

    X_test = np.linspace(-1.2, 1.2, N_test)

    return X[:, None], Y[:, None], X_test[:, None]


class SCurveDataset(DensityDataset):
    def _create_data(self):
        data, _ = datasets.make_s_curve(
            n_samples=self.n_samples, noise=self.noise, random_state=self.seed
        )
        data = data[:, [0, 2]]
        self.data = data


class BlobsDataset(DensityDataset):
    def _create_data(self):
        data, _ = datasets.make_blobs(n_samples=self.n_samples, random_state=self.seed)
        self.data = data


class MoonsDataset(DensityDataset):
    def _create_data(self):
        data, _ = datasets.make_moons(
            n_samples=self.n_samples, noise=self.noise, random_state=self.seed
        )
        self.data = data


class SwissRollDataset(DensityDataset):
    def _create_data(self):
        data, _ = datasets.make_swiss_roll(
            n_samples=self.n_samples, noise=self.noise, random_state=self.seed
        )
        data = data[:, [0, 2]]
        self.data = data


class NoisySineDataset(DensityDataset):
    def _create_data(self):

        rng = np.random.RandomState(seed=self.seed)
        x = np.abs(2 * rng.randn(1, self.n_samples))
        y = np.sin(x) + 0.25 * rng.randn(1, self.n_samples)
        self.data = np.vstack((x, y)).T


class CheckBoard(DensityDataset):
    def _create_data(self):
        rng = np.random.RandomState(self.seed)
        x1 = rng.rand(self.n_samples) * 4 - 2
        x2_ = rng.rand(self.n_samples) - rng.randint(0, 2, self.n_samples) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
        self.data = data + 0.001 * rng.randn(*data.shape)
