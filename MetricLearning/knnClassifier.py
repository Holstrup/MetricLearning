from Model import get_data
import numpy as np
from statistics import mode

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
    return mode(results)

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
        Distances[i] = chi_square_distance(L @ xi, X[i, :] @ L)
    return Distances