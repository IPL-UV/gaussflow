import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".here"])

# append to path
sys.path.append(str(root))

# @title Load Packages
# TYPE HINTS
from typing import Tuple, Optional, Dict, Callable, Union

# PyTorch Settings
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.distributions as dist

# PyTorch Lightning Settings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from src.data.mnist_sklearn import discretize, SklearnDigits
from torchvision.transforms import Compose

from argparse import ArgumentParser

# NUMPY SETTINGS
import numpy as np

np.set_printoptions(precision=3, suppress=True)

# MATPLOTLIB Settings
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt

# SEABORN SETTINGS
import seaborn as sns

sns.set_context(context="talk", font_scale=0.7)

from src.experiments.mnist_sklearn.model import (
    add_simple_mnist_model_args,
    create_simple_mnist_model,
)
from src.lit_image import ImageFlow


# LOGGING SETTINGS
import ml_collections
import wandb

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device: {}".format(device))


def cli_main():

    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ImageFlow.add_model_specific_args(parser)
    parser = SklearnDigits.add_ds_args(parser)
    parser = add_simple_mnist_model_args(parser)
    args = parser.parse_args()

    # wandb_logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity)
    # wandb_logger.experiment.config.update(args)

    # ------------
    # data
    # ------------
    transforms = Compose([torch.Tensor, discretize])

    train_ds = SklearnDigits(mode="train", image=True, transforms=transforms)
    valid_ds = SklearnDigits(mode="valid", image=True, transforms=transforms)
    test_ds = SklearnDigits(mode="test", image=True, transforms=transforms)

    args.n_total_steps = args.max_epochs * len(train_ds)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # ------------
    # model
    # ------------
    X_init = train_ds[:16]

    inn = create_simple_mnist_model(
        img_shape=(1, 8, 8),
        X_init=X_init,
        n_subflows=args.n_subflows,
        actnorm=args.actnorm,
        n_reflections=args.n_reflections,
        mask=args.mask,
    )

    flow_img_mnist = ImageFlow(inn, cfg=args, prior=None)

    # Test
    z, log_jac_det = flow_img_mnist.model.forward(X_init)

    x_ori, log_jac_det = flow_img_mnist.model.forward(z, rev=True)

    samples = flow_img_mnist.sample((16, 64))
    assert X_init.shape == samples.shape

    # ========================
    # TRAINING
    # ========================
    trainer = pl.Trainer(
        # epochs
        min_epochs=5,
        max_epochs=args.max_epochs,
        # progress bar
        progress_bar_refresh_rate=100,
        # device
        gpus=0,
        # gradient norm
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        # logger=wandb_logger,
    )

    trainer.fit(
        flow_img_mnist,
        train_dataloader=train_dl,
        val_dataloaders=valid_dl,
    )

    # ------------
    # testing
    # ------------
    test_result = trainer.test(flow_img_mnist, test_dataloaders=test_dl)


if __name__ == "__main__":
    cli_main()
