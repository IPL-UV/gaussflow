import torch
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
import wandb
import corner
import numpy as np

from src.data.utils import tensor2numpy


class LogEvalImages(pl.Callback):
    def __init__(
        self, input_imgs: torch.Tensor, every_n_epochs: int = 1, n_features: int = 10
    ):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.n_features = n_features

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:

            # =======================
            # plot Gaussianized data
            # =======================
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():

                z, _ = pl_module.model.forward(input_imgs, rev=False)
                fig = corner.corner(tensor2numpy(z)[:, : self.n_features])

                fig_gaussianzied = wandb.Image(fig)

            # =======================
            # Plot generated images
            # =======================
            # get samples
            with torch.no_grad():
                pl_module.eval()
                samples = pl_module.sample(z.shape)
                pl_module.train()

                # corner plot
                img_grid = make_grid(
                    samples[:64] / 255,
                    nrow=8,
                    ncol=8,
                    padding=2,
                )
                # convert to wandb type
                save_image(img_grid, "./temp_img.png")

                fig_samples = wandb.Image("./temp_img.png")

            # create table
            my_data = [[fig_gaussianzied, fig_samples]]

            # create a wandb.Table() with corresponding columns
            columns = ["latent", "samples"]
            my_table = wandb.Table(data=my_data, columns=columns)
            wandb.log({f"epoch_{pl_module.current_epoch}": my_table})
