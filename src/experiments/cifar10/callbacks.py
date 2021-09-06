from pytorch_lightning.callbacks import Callback
import corner
import torch
import wandb


class LogEvalImagesCallback(Callback):
    """Class to log a fixed set of image to monitor input and reconstructed image along with the loss for that particular sample.

    Args:
        Callback (pytorch_lightning.callbacks.Callbacks): Inheriting from this parent class.
    """

    def __init__(
        self,
        test_data,
        img_log_freq=100,
        num_x: int = 6,
        num_y: int = 6,
        n_features: int = 10,
    ):
        self.test_data = test_data
        self.img_log_freq = img_log_freq
        self.num_x = num_x
        self.num_y = num_y
        self.n_features = n_features

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Called when the val epoch begins.
        Logs the images in a table every 'img_log_freq' epochs.
        """

        if pl_module.current_epoch % self.img_log_freq == 0:

            # get samples
            samples = pl_module.get_samples()

            fig, ax = plot_images(samples, self.num_x, self.num_y)

            fig_samples = wandb.Image(plt)

            # plot Gaussianized data
            with torch.no_grad():
                z, _ = pl_module.model.forward(self.test_data, rev=False)
                fig = corner.corner(z.detach().numpy()[:, : self.n_features])

            fig_gaussianzied = wandb.Image(fig)

            # create table
            my_data = [fig_gaussianzied, fig_samples]

            # create a wandb.Table() with corresponding columns
            columns = ["latent", "samples"]
            test_table = wandb.Table(data=my_data, columns=columns)
            wandb.log({f"epoch_{pl_module.current_epoch}": test_table})


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context(context="talk", font_scale=0.7)


def plot_images(data, num_x, num_y):
    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = data[i]

        if data.ndim == 2:
            plottable_image = np.reshape(plottable_image, (3, 32, 32))

        ax.imshow(plottable_image.transpose([1, 2, 0]), cmap="gray")
        ax.axis("off")
    return fig, ax
