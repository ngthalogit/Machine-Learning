import numpy as np

# functions for distance computations
def dist_pp(z, x): # distance between point and point
    return np.sum((z - x) * (z - x))

def dist_ps_faster(z, X): # distance between point to set of points following the formula
   z2 = np.sum(z * z)
   X2 = np.sum(X * X, 1)
   return z2 + X2 - 2 * np.dot(X, z)

def dist_ss(Z, X): # distance between set to set
    Z2 = np.sum(Z * Z, 1)
    X2 = np.sum(X * X, 1)
    return  Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2 * np.dot(Z, X.T)



