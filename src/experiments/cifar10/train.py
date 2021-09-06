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
from src.experiments.cifar10.models import get_model_architecture, get_model_args
from torchvision.transforms import Compose
from pl_bolts.datamodules import CIFAR10DataModule, TinyCIFAR10DataModule

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

from src.experiments.cifar10.data import (
    add_cifar10_ds_args,
)
from src.experiments.cifar10.models.glow import (
    create_multiscale_cifar10_model_permute,
    add_multiscale_cifar10_model_args,
)
from src.experiments.cifar10.trainer import ImageFlow


# LOGGING SETTINGS
import ml_collections
import wandb
from src.experiments.utils import add_wandb_args, update_args_yaml

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device: {}".format(device))


def cli_main():

    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    # model specific arguments
    parser = ImageFlow.add_model_specific_args(parser)

    # Dataset Arguments
    parser = add_cifar10_ds_args(parser)

    # Logger arguments
    parser = add_wandb_args(parser)

    # Model Arguments
    parser = get_model_args(parser, "gf")

    # Trainer-Specific Arguments
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    wandb_logger = WandbLogger(
        config=args, project=args.wandb_project, entity=args.wandb_entity
    )

    # ------------
    # data
    # ------------
    import torchvision

    def discretize(sample):
        return (sample * 255).to(torch.int32)

    train_transforms = torchvision.transforms.Compose(
        [
            # torchvision.transforms.RandomCrop(32, padding=4),
            # torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            discretize,
        ]
    )
    test_transforms = torchvision.transforms.Compose(
        [
            # torchvision.transforms.RandomCrop(32, padding=4),
            # torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            discretize,
        ]
    )

    cifar10_dm = CIFAR10DataModule(
        data_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        val_transforms=test_transforms,
    )
    # cifar10_dm = TinyCIFAR10DataModule(
    #     data_dir=args.dataset_dir,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     num_samples=100,
    #     train_transforms=train_transforms,
    #     test_transforms=test_transforms,
    #     val_transforms=test_transforms,
    # )

    cifar10_dm.prepare_data(download=True)
    cifar10_dm.setup()

    args.n_total_steps = args.max_epochs * cifar10_dm.num_samples

    # train_dl = DataLoader(
    #     train_ds,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     pin_memory=True,
    #     num_workers=args.num_workers,
    # )
    # valid_dl = DataLoader(
    #     valid_ds,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    # )
    # test_dl = DataLoader(
    #     test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    # )

    # ------------
    # model
    # ------------
    train_dl = cifar10_dm.train_dataloader()
    for ix, iy in train_dl:

        break
    X_init = ix

    inn = get_model_architecture(args, X_init, model="gf")

    flow_img_cifar10 = ImageFlow(inn, cfg=args, prior=None)

    # # Test
    z, log_jac_det = flow_img_cifar10.model.forward(X_init)

    x_ori, log_jac_det = flow_img_cifar10.model.forward(z, rev=True)

    samples = flow_img_cifar10.sample((64, 768))
    assert X_init.shape == samples.shape

    # ========================
    # TRAINING
    # ========================
    trainer = pl.Trainer(
        # epochs
        min_epochs=2,
        max_epochs=args.max_epochs,
        # progress bar
        progress_bar_refresh_rate=100,
        # device
        gpus=args.gpus,
        # gradient norm
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        logger=wandb_logger,
    )

    trainer.fit(
        flow_img_cifar10,
        datamodule=cifar10_dm,
    )

    # # ------------
    # # testing
    # # ------------
    test_result = trainer.test(flow_img_cifar10, datamodule=cifar10_dm)


if __name__ == "__main__":
    cli_main()
