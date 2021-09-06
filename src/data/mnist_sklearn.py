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
from einops import rearrange, repeat
import torch
from argparse import ArgumentParser


class SklearnDigits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(
        self, mode: str = "train", image: bool = True, transforms: bool = None
    ):
        digits = load_digits()

        # change shape
        if image:
            digits = digits.images
            digits = repeat(
                digits,
                "B H W -> B C H W",
                C=1,
            )
        else:
            digits = digits.data

        if mode == "train":
            self.data = digits[:1000]
        elif mode == "val":
            self.data = digits[1000:1350]
        else:
            self.data = digits[1350:]

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    @staticmethod
    def add_ds_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # data args
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=4)
        return parser


def discretize(sample):
    return (sample / 16.0 * 255).to(torch.int32)
