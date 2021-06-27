import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import wandb

sns.set_context(context="talk", font_scale=0.7)


def plot_2d_joint(
    data,
    color="blue",
    kind="kde",
    save=None,
    wandb_logger=None,
    log_name="",
    **kwargs,
):

    plt.figure(figsize=(5, 5))

    sns.jointplot(
        x=data[:, 0],
        y=data[:, 1],
        kind=kind,
        color=color,
        **kwargs,
    )
    plt.tight_layout()
    if wandb_logger is not None:
        wandb_logger.log({f"{log_name}": wandb.Image(plt)})
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def plot_2d_joint_probs(
    data,
    probs,
    save=None,
    wandb_logger=None,
    log_name="",
    **kwargs,
):
    # Plot the Probabilities of the data using colors
    fig, ax = plt.subplots()
    g = ax.scatter(data[:, 0], data[:, 1], s=1, c=probs, **kwargs)
    ax.set_title("Probabilities")
    plt.colorbar(g)
    plt.tight_layout()
    if wandb_logger is not None:
        wandb_logger.log({f"{log_name}": wandb.Image(plt)})
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def generate_2d_grid(data: np.ndarray, n_grid: int = 1_000, buffer: float = 0.01) -> np.ndarray:

    xline = np.linspace(data[:, 0].min() - buffer, data[:, 0].max() + buffer, n_grid)
    yline = np.linspace(data[:, 1].min() - buffer, data[:, 1].max() + buffer, n_grid)
    xgrid, ygrid = np.meshgrid(xline, yline)
    xyinput = np.concatenate([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], axis=1)
    return xyinput

