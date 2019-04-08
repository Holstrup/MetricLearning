import matplotlib.pyplot as plt
import numpy as np


def plot_some_data(embedding_list, label_list):
    """
    Outputs a plot of a few of the embeddings in histogram form.

    :param N_SUBPLOTS: Number of subplots in plot
    """
    N_plots, DIMENSIONS = np.shape(embedding_list)
    fig = plt.figure()
    y_max = np.max(embedding_list) + 0.02
    for i in range(N_plots):
        ax = fig.add_subplot(N_plots, 1, i + 1)
        label = label_list[i]
        ax.set_title("{}".format(label), size=12)
        ax.bar(list(range(DIMENSIONS)), embedding_list[i, :])
        ax.set_ylim([0, y_max])
    plt.subplots_adjust(wspace=0.1, hspace = 0.6)
    plt.show()


def plot_kernel(L):
    plt.imshow(L)
    plt.colorbar()
    plt.show()


def plot_loss_curve(losses):
    plt.plot(losses)
    plt.show()


