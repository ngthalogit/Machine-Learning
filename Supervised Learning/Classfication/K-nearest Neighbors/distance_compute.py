"""
This file define the function computing the Euclid distance

"""

import numpy as np


# distance between point and point
def dist_pp(z, x):
    return np.sum((z - x) * (z - x))


# distance between point to set
def dist_ps_faster(z, X):
    z2 = np.sum(z * z)
    X2 = np.sum(X * X, 1)
    return z2 + X2 - 2 * np.dot(X, z)


# distance between set to set
def dist_ss(Z, X):
    Z2 = np.sum(Z * Z, 1)
    X2 = np.sum(X * X, 1)
    return Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2 * np.dot(Z, X.T)
