# spyder up to find the root
from pathlib import Path
from typing import Optional, Tuple, NamedTuple

import numpy as np
import pytorch_lightning as pl
from pyprojroot import here
from skimage.transform import downscale_local_mean
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

root = here(project_files=[".here"])

class HSIShape(NamedTuple):
    channels: int
    width: int
    height: int


class HSIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 128,
        val_split: float = 0.2,
        seed: int = 123,
        flatten: bool = True,
        subset: Optional[int] = None,
        dataset_dir: str = None,
        downscale_factor: Optional[int] = None,
    ):

        self.val_split = val_split
        self.batch_size = batch_size
        self.seed = seed
        self.flatten = flatten
        self.subset = subset
        if dataset_dir is None:
            dataset_dir = str(Path(root).joinpath("datasets/hsi"))
        self.dataset_dir = dataset_dir
        self.downscale_factor = downscale_factor

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):

        # assign train/val split
        Xtrain, Xval = train_test_split(
            self.train_dataset.data, test_size=self.val_split, random_state=self.seed,
        )

        Xtrain = Xtrain.numpy().astype(np.float32)[..., None]
        Xval = Xval.numpy().astype(np.float32)[..., None]

        if self.subset is not None:
            Xtrain = Xtrain[: self.subset]

        if self.downscale_factor:
            Xtrain = downscale_local_mean(
                Xtrain, (1, self.downscale_factor, self.downscale_factor, 1)
            )
            _, H, W, C = Xtrain.shape
            self.image_shape = HSIShape(channels=C, height=H, width=W)
            Xval = downscale_local_mean(
                Xval, (1, self.downscale_factor, self.downscale_factor, 1)
            )

        if self.flatten:
            Xtrain = flatten_image(Xtrain, self.image_shape, batch=True)
            Xval = flatten_image(Xval, self.image_shape, batch=True)

        self.Xtrain = Xtrain
        self.Xval = Xval
        self.ds_train = GenericDataset(Xtrain)
        self.ds_val = GenericDataset(Xval)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def valid_dataloader(self):
        return DataLoader(
            self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

