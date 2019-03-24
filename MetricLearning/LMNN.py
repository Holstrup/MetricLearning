import numpy as np


"""
Hyperparameters
 
"""
l = 1    # Margin Parameter
mu = 1   # Trade-off between push & pull
K = 2    # K Target Neighbors


"""
Functions

"""

def chi_square_distance(xi, xj):
    """
    Chi square distance

    :param xi: Embedding (1, F)
    :param xj: Target Neighbor (1, F)
    :return: Distance
    """
    return 1/2 * np.sum(np.square(xi-xj) / (xi + xj))


def loss_function(xi, xj, xk):
    """
    Loss function as described in the paper

    :param xi: One embedding        (1, F)
    :param xj: K target neighbors   (K, F)
    :param xk: Unknown imposters    (?, F)

    :return: Loss for one embedding
    """

    _, K = np.shape(xj)
    imposter, _ = np.shape(xj)
    sum = 0

    for j in range(K):
        sum += chi_square_distance(L @ xi, L @ xj[:, j])
        inner_sum = 0
        for k in range(imposter):
            inner_sum += max(0, l + chi_square_distance(xi, xj[:, j]) - chi_square_distance(xi, xk[k, :]))
        sum += mu * inner_sum
    return sum


def distance(xi, X):
    """
    :param xi: Embedding vector (1, F)
    :param X: Data matrix without embedding vector (N, F)
    :return: Distance vector (1, N)
    """
    return np.linalg.norm((X - xi), axis=1)


def find_triplets(xi, yi, X, y):
    """
    Given some vector xi and corresponding label yi, find target neighbors and imposters

    :param xi: Embedding vector (1, F)
    :param yi: Label for embedding vector
    :param X: Full data matrix  (N, F)
    :return: target_neighbors and imposters for embedding (K, F) (?, F)
    """
    candidate_target_neighbors = X[np.where(yi == y)]
    imposters = X[np.where(yi != y)]

    target_neighbors_dist = distance(xi, candidate_target_neighbors)
    imposters_dist = distance(xi, imposters)

    # Find K target neighbors
    target_neighbors = np.zeros((1, F))
    for i in range(K):
        min_index = np.argmin(target_neighbors_dist)
        target_neighbors = np.vstack((target_neighbors, candidate_target_neighbors[min_index]))
        candidate_target_neighbors = np.delete(candidate_target_neighbors, (min_index), axis=0)
    target_neighbors = target_neighbors[1:, :]

    # Find ? imposters
    max_target_dist = np.max(target_neighbors_dist)
    imposters = imposters[np.where(imposters_dist < max_target_dist + l)]

    return target_neighbors, imposters
