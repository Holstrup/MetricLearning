from Model import get_data
import matplotlib.pyplot as plt


def plot_some_data(N_SUBPLOTS=3):
    """
    Outputs a plot of a few of the embeddings in histogram form.

    :param N_SUBPLOTS: Number of subplots in plot
    """
    embeddings, labels = get_data()
    fig = plt.figure()
    for i in range(N_SUBPLOTS):
        ax = fig.add_subplot(N_SUBPLOTS, 1, i + 1)
        label = labels[5 * i]
        ax.set_title("{}".format(label), size=12)
        ax.bar(list(range(2048)), embeddings[:, 5 * i])
    plt.show()
