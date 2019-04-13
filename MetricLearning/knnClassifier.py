from Model import get_data
import numpy as np
from collections import defaultdict

def knn(embedding, matrix_embeddings, labels, L, k):
    """
    Function that returns the nearest neighbor to an image encoding

    :param encodings: Vector of encoding for an image (100, 40)
    :return: Predicted label
    """

    dist_vector = distance(embedding, matrix_embeddings, L)
    closest_indices = np.argsort(dist_vector)[0:k]

    results = []
    for index in closest_indices:
        results.append(labels[index])
    return results

def chi_square_distance(xi, xj):
    """
    Chi square distance

    :param xi: Embedding       (1, D)
    :param xj: Target Neighbor (1, D)
    :return: Distance
    """
    return 1 / 2 * np.nansum(np.square(xi - xj) / (xi + xj))

def distance(xi, X, L):
    """
    :param xi: Embedding vector                        (1, F)
    :param X: Data matrix without embedding vector (N - 1, F)
    :return: Distance vector                       (1, N - 1)
    """
    N, D = np.shape(X)
    Distances = np.zeros(N)
    for i in range(N):
        Distances[i] = chi_square_distance(L @ xi, L @ X[i, :])
    return Distances

#In case more elements have the same number of occurances, a random one out of them is chosen
def calc_mode(*args, freq_tab: dict = None, **kwargs) -> tuple:
    def mode():
        highest_frequency = max(mode.frequencies.values())
        m = [str(e) for e, f in mode.frequencies.items() if f == highest_frequency]
        m = tuple(sorted(m))

        if not 1 <= len(m) <= mode.max_modes:
            m = None
        return m

    mode.max_modes = 100

    if freq_tab is not None:
        mode.frequencies = freq_tab
    elif len(kwargs) > 0:
        mode.frequencies = kwargs
    else:
        freq = defaultdict(int)
        elems = args[0] if len(args) == 1 and type(args[0]) == list else args
        for e in elems:
            freq[e] += 1
        mode.frequencies = freq

    return mode()