import matplotlib.pyplot as plt
import seaborn as sns


def plot_2d_joint(data, color="blue", kind="kde", title="Original Data", **kwargs):

    fig = plt.figure(figsize=(5, 5))

    sns.jointplot(x=data[:, 0], y=data[:, 1], kind=kind, color=color, **kwargs)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
