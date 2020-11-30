import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

sns.set_context(context="talk", font_scale=0.7)


def plot_data(x, bandwidth=0.2, **kwargs):
    kde = stats.gaussian_kde(x[:, 0])
    x_axis = np.linspace(-5, 5, 200)
    plt.plot(x_axis, kde(x_axis), **kwargs)


def plot_hist(x, save=None, **kwargs):
    fig, ax = plt.subplots()
    ax.hist(x, bins=100, **kwargs)
    plt.legend()
    if save is not None:
        fig.savefig(save)
    else:
        plt.show()