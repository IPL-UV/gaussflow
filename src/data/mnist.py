# spyder up to find the root
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
from pyprojroot import here
from skimage.transform import downscale_local_mean
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from sklearn.datasets import load_digits
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

# from rbig_jax.custom_types import ImageShape
# from rbig_jax.data import GenericDataset
# from rbig_jax.transforms.reshape import flatten_image, unflatten_image

# root = here(project_files=[".here"])

# image_shape = ImageShape(C=1, H=28, W=28)


class SklearnDigits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(
        self, mode: str = "train", image: bool = True, transforms: bool = None
    ):
        digits = load_digits()

        # change type
        digits = digits.astype(np.float32)

        # change shape
        if image:
            digits = rearrange(digits, "B (C W H) -> B C W H", C=1, H=8, W=8)

        if mode == "train":
            self.data = digits.data[:1000]
        elif mode == "val":
            self.data = digits.data[1000:1350]
        elif mode == "test":
            self.data = digits.data[1350:]
        else:
            raise ValueError(f"Unrecognized ds mode.")

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample


# class MNISTDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         batch_size: int = 128,
#         val_split: float = 0.2,
#         seed: int = 123,
#         flatten: bool = True,
#         subset: Optional[int] = None,
#         dataset_dir: str = None,
#         downscale_factor: Optional[int] = None,
#     ):

#         self.val_split = val_split
#         self.batch_size = batch_size
#         self.seed = seed
#         self.flatten = flatten
#         self.image_shape = ImageShape(C=1, H=28, W=28)
#         self.subset = subset
#         if dataset_dir is None:
#             dataset_dir = str(Path(root).joinpath("datasets/mnist"))
#         self.dataset_dir = dataset_dir
#         self.downscale_factor = downscale_factor

#     def prepare_data(self):
#         # download
#         self.train_dataset = MNIST(self.dataset_dir, download=True, train=True)

#         # self.test_dataset = MNIST(os.getcwd(), download=True, train=False)

#     def setup(self, stage=None):

#         # assign train/val split
#         Xtrain, Xval = train_test_split(
#             self.train_dataset.data,
#             test_size=self.val_split,
#             random_state=self.seed,
#         )

#         Xtrain = Xtrain.numpy().astype(np.float32)[..., None]
#         Xval = Xval.numpy().astype(np.float32)[..., None]

#         if self.subset is not None:
#             Xtrain = Xtrain[: self.subset]

#         if self.downscale_factor:
#             Xtrain = downscale_local_mean(
#                 Xtrain, (1, self.downscale_factor, self.downscale_factor, 1)
#             )
#             _, H, W, C = Xtrain.shape
#             self.image_shape = ImageShape(C=C, H=H, W=W)
#             Xval = downscale_local_mean(
#                 Xval, (1, self.downscale_factor, self.downscale_factor, 1)
#             )

#         if self.flatten:
#             Xtrain = flatten_image(Xtrain, self.image_shape, batch=True)
#             Xval = flatten_image(Xval, self.image_shape, batch=True)

#         self.Xtrain = Xtrain
#         self.Xval = Xval
#         self.ds_train = GenericDataset(Xtrain)
#         self.ds_val = GenericDataset(Xval)

#     def train_dataloader(self):
#         return DataLoader(
#             self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=0
#         )

#     def valid_dataloader(self):
#         return DataLoader(
#             self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=0
#         )
