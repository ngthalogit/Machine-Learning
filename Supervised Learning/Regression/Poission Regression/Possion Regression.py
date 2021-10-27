import numpy as np


X = []
y = []


def loss(x, y, w):
    M = x.shape[0]
    lamd = np.exp(np.dot(w.T, x)) # y_hat
    error = (lamd - np.dot(w.T, x) * y) / M
    return error


def derivative(x, y, w):
    lamd = np.exp(np.dot(w.T, x))
    return np.dot(x.T, (lamd - y))


def fit(x, y, w, iteration=1000, lr=0.01):
    for i in iteration:
        w = w - derivative(x, y, w) * lr
        w_old = w
        hist[i] = loss(x, y, w)
    return w, hist