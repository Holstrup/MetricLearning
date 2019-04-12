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