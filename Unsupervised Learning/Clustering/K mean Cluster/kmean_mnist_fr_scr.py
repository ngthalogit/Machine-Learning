# lib
from mnist import MNIST
import numpy as np
from distance_compute import dist_ss

# constants
K = 10
label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DIR = 'C:\\Users\ADMIN\PycharmProjects\MLprojects\Kmean\dataset'

# init dataset
mnist = MNIST(DIR)
x_train, y_train = mnist.load_training()  # 60000 samples
x_test, y_test = mnist.load_testing()  # 10000 samples

X_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)
X_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.int32)


def init_centroids(X, k):
    N = X.shape[0]
    index = np.random.choice(np.arange(N), k, replace=False)
    return X[index]


def labling_centroids(centroids, X):
    dis = dist_ss(centroids, X)
    return np.argmin(dis, axis=0)


def update_centroids(X, labels, k):
    cent = np.ones((k, X.shape[1]))
    for i in range(k):
        Xk = X[labels == i, :]
        cent[i, :] = np.mean(Xk, axis=0)
    return cent


def is_equal(centroids_A, centroids_B):
    A = [tuple(a) for a in centroids_A]
    B = [tuple(b) for b in centroids_B]
    return (A == B)


def fit(X, k):
    centroids = init_centroids(X, k)
    labels = None
    while True:
        labels = labling_centroids(centroids, X)
        updated_centroids = update_centroids(X, labels, k)
        if is_equal(centroids, updated_centroids):
            break
        centroids = updated_centroids
    return centroids


res = fit(X_train, K)

y_pred = labling_centroids(res, X_test)

# TODO - optimize the problem with centroids initialization algorithm
