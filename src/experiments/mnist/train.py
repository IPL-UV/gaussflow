#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/IPL-UV/gaussflow/blob/master/docs/assets/demo/pytorch_nf_freia.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # PyTorch PlayGround
#
# This is my notebook where I play around with all things PyTorch. I use the following packages:
#
# * PyTorch
# * Pyro
# * GPyTorch
# * PyTorch Lightning
#

# In[1]:


# @title Install Packages
# %%capture
import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".here"])

# append to path
sys.path.append(str(root))

# In[2]:


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
# sns.set(rc={'figure.figsize': (12, 9.)})
# sns.set_style("whitegrid")

# PANDAS SETTINGS
import pandas as pd

pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)

# LOGGING SETTINGS
import ml_collections
import wandb

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device: {}".format(device))

# torch.multiprocessing.set_start_method("spawn", force=True)

cfg = ml_collections.ConfigDict()

# Logger
cfg.wandb_project = "cvpr2021"
cfg.dataset = "mnist_28x28"
cfg.wandb_entity = "ipl_uv"

cfg.batch_size = 256
cfg.num_workers = 12
cfg.seed = 123

# Data
cfg.n_train = 55_000
cfg.n_valid = 5_000
cfg.n_test = 10_000
cfg.noise = 0.05
cfg.n_total_dims = 1 * 28 * 28

# Model
cfg.loss_fn = "bpd"
cfg.n_layers = 8
cfg.multiscale = False
cfg.n_reflections = 10
cfg.n_bins = 8
cfg.model = "rnvp"
cfg.actnorm = True

# Training
cfg.num_epochs = 100
cfg.lr_scheduler = None
cfg.weight_decay = 1e-4
cfg.gamma = 0.99
cfg.learning_rate = 1e-4
cfg.n_total_steps = cfg.num_epochs * cfg.n_train

# Testing
cfg.importance_samples = 8
cfg.temperature = 1.0


# In[4]:


seed_everything(cfg.seed)


# ## Logging

# In[5]:


wandb_logger = WandbLogger(project=cfg.wandb_project, entity=cfg.wandb_entity)
wandb_logger.experiment.config.update(cfg)


# ## Data

# In[6]:


from pl_bolts.datamodules import MNISTDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.mnist import MNIST
from torchvision import transforms


# In[7]:


# mnist_dm = MNISTDataModule(
#     data_dir="/datadrive/eman/datasets/mnist/",
#     num_workers=4,
#     normalize=False,
#     batch_size=64,
#     seed=123,
#     transforms=train_transforms


# )
# mnist_dm.prepare_data(download=False)
# mnist_dm.setup()


# In[8]:


def discretize(sample):
    return (sample * 255).to(torch.int32)


train_transforms = transforms.Compose(
    [
        # torchvision.transforms.RandomCrop(32, padding=4),
        # torchvision.transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        discretize,
    ]
)

dataset = MNIST(
    "/datadrive/eman/datasets/mnist/",
    train=True,
    download=True,
    transform=train_transforms,
)
mnist_test = MNIST(
    "/datadrive/eman/datasets/mnist/",
    train=False,
    download=True,
    transform=train_transforms,
)
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(
    mnist_train,
    shuffle=True,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    mnist_val,
    shuffle=False,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    pin_memory=True,
)
test_loader = DataLoader(
    mnist_test,
    shuffle=False,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    pin_memory=True,
)

# mnist_dm = MNISTDataModule(
#     data_dir="/datadrive/eman/datasets/mnist/",
#     num_workers=12,
#     normalize=False,
#     batch_size=256,
#     seed=123,
# )
# mnist_dm.prepare_data(download=False)
# mnist_dm.setup()

# train_loader = mnist_dm.train_dataloader()
# val_loader = mnist_dm.val_dataloader()
# test_loader = mnist_dm.test_dataloader()


# In[9]:


# In[11]:


X_init = []

for i, (ix, iy) in enumerate(train_loader):

    if isinstance(X_init, list):
        X_init = ix
    else:
        X_init = torch.cat([X_init, ix], dim=0)

    if X_init.shape[0] > 1_000:
        break


# In[14]:


from torchvision.utils import make_grid


def visualize_images(input_imgs):

    grid = make_grid(input_imgs.to(torch.int32), nrow=8, ncols=8)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7, 7))
    plt.title(f"Images")
    plt.imshow(grid)
    plt.axis("off")
    plt.show()


# #### DataLoader

# In[15]:


# ## Model

# ## Normalizing Flow Models

# In[16]:


# # FrEIA imports
# import FrEIA.framework as Ff
# import FrEIA.modules as Fm
# from src.models.layers.dequantization import UniformDequantization
# from src.models.layers.convolutions import Conv1x1, Conv1x1Householder, ConvExponential
# from src.models.layers.multiscale import SplitPrior

# # #### Coupling Network

# # In[17]:


# # subset net
# def subnet_conv(c_in, c_out):
#     return nn.Sequential(
#         nn.Conv2d(c_in, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, c_out, 3, padding=1)
#     )


# # REALNVP Model
# from src.models.flows.realnvp import append_realnvp_coupling_block_image

# n_channels = 1
# height = 28
# width = 28

# inn = Ff.SequenceINN(n_channels, height, width)


# # uniform dequantization (for integers)
# inn.append(UniformDequantization, num_bits=8)

# print("Input:")
# print(X_init.shape, np.prod(inn(X_init)[0].shape))

# inn.append(Fm.Flatten)
# print(inn(X_init)[0].shape, np.prod(inn(X_init)[0].shape))


# def subnet_fc(c_in, c_out):
#     return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(), nn.Linear(512, c_out))


# for k in range(12):
#     inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc)


# # subset net
# def subnet_conv(c_in, c_out):
#     return nn.Sequential(
#         nn.Conv2d(c_in, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, c_out, 3, padding=1)
#     )


# inn.append(
#     Fm.IRevNetDownsampling,
# )

# print(inn(X_init)[0].shape, np.prod(inn(X_init)[0].shape))


# for isubflow in range(8):

#     # append RealNVP Coupling Block
#     inn = append_realnvp_coupling_block_image(
#         inn,
#         conditioner=subnet_conv,
#         actnorm=False,
#         n_reflections=None,
#         mask=None,
#         permute=True,
#     )

# # inn.append(Fm.Flatten)
# print("Final:")
# print(inn(X_init)[0].shape)

from src.experiments.mnist.models.glow import (
    get_fc_glow_model,
    get_multiscale_glow_model,
)
from src.experiments.mnist.models.realnvp import get_multiscale_realnvp_model
from src.experiments.mnist.models.gf import get_gf_fc_model

# inn, output_shape = get_fc_glow_model(X_init, n_layers=12, n_hidden=512, quantization=True)
# inn, output_dim = get_multiscale_realnvp_model(
#     X_init, quantization=True, mask="checkerboard"
# )
# inn, output_dim = get_multiscale_glow_model(
#     X_init, quantization=True, mask="checkerboard"
# )
inn, output_dim = get_gf_fc_model(X_init=X_init, n_layers=20, non_linear="gaussian")


# Image Flow Model

from src.lit_image import ImageFlow


# In[38]:


# Training
cfg.num_epochs = 200
cfg.lr_scheduler = "cosine_annealing"
cfg.weight_decay = 1e-4
cfg.gamma = 0.99
cfg.learning_rate = 5e-4
cfg.n_total_steps = cfg.num_epochs * 55_000


# In[39]:


flow_img_mnist = ImageFlow(inn, cfg=cfg, prior=None)


# ### Forward

# In[40]:


with torch.no_grad():
    z, log_jac_det = flow_img_mnist.model.forward(X_init)
print(z.shape)


# In[41]:


fig = corner.corner(z.cpu().numpy()[:, :10])


# #### Inverse

# In[42]:


with torch.no_grad():

    x_ori, log_jac_det = flow_img_mnist.model.forward(z, rev=True)
print(x_ori.shape, x_ori.min(), x_ori.max())


# In[43]:


visualize_images(x_ori)


# #### Samples

# In[44]:


# sample from the INN by sampling from a standard normal and transforming
# it in the reverse direction
n_samples = 64
# z = torch.randn(n_samples, N_DIM)
with torch.no_grad():
    samples = flow_img_mnist.sample((n_samples, output_dim))

# plot_digits(samples.detach().numpy().squeeze(), 4, 4)


# In[45]:


visualize_images(samples)


# ## Training

# In[46]:


from src.callbacks.images import LogEvalImages

trainer = pl.Trainer(
    # epochs
    min_epochs=1,
    max_epochs=cfg.num_epochs,
    # progress bar
    progress_bar_refresh_rate=10,
    # device
    gpus="1",
    # gradient norm
    gradient_clip_val=10.0,
    gradient_clip_algorithm="norm",
    logger=wandb_logger,
    callbacks=[LogEvalImages(input_imgs=X_init, every_n_epochs=5)],
)

trainer.fit(flow_img_mnist, train_dataloader=train_loader, val_dataloaders=val_loader)
# trainer.fit(flow_img_mnist, datamodule=mnist_dm)

result = trainer.test(flow_img_mnist, test_dataloaders=test_loader)
result = trainer.test(flow_img_mnist, test_dataloaders=val_loader)


# # #### Latent Space

# # In[117]:


# x = torch.Tensor(X_init)
# z, log_jac_det = flow_img_mnist.model(x)

# # plot_digits(z.detach().numpy(), 4, 4)


# # In[ ]:


# # In[118]:


# fig = corner.corner(z.detach().numpy()[:, :5], hist_factor=2, color="red")


# # #### Inverse Transform

# # In[119]:


# x_ori, _ = flow_img_mnist.model.forward(z, rev=True)

# plot_digits(x_ori.detach().numpy().squeeze(), 4, 4)


# # In[120]:


# x_ori.shape


# # In[105]:


# fig = corner.corner(
#     x_ori.detach().numpy()[:, :5, ...].flatten(), hist_factor=2, color="green"
# )


# # #### Sampling

# # In[121]:


# # sample from the INN by sampling from a standard normal and transforming
# # it in the reverse direction
# n_samples = 16
# # z = torch.randn(n_samples, N_DIM)
# samples = flow_img_mnist.sample((100, 64))

# plot_digits(samples.detach().numpy().squeeze(), 4, 4)


# # In[ ]:


# fig = corner.corner(samples.detach().numpy(), hist_factor=2, color="red")
