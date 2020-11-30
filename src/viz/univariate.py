import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_data(x, bandwidth=0.2, **kwargs):
    kde = stats.gaussian_kde(x[:, 0])
    x_axis = np.linspace(-5, 5, 200)
    plt.plot(x_axis, kde(x_axis), **kwargs)


def plot_hist(x, **kwargs):
    fig, ax = plt.subplots()
    ax.hist(x, bins=100, **kwargs)
    plt.legend()
    plt.show()