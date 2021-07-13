"""

auth: Nguyen Thanh Long

"""

import numpy as np


def fit(X_bias, y):
    XX = np.linalg.pinv(np.dot(X_bias, X_bias.T))
    XXX = np.dot(XX, X_bias)
    w = np.dot(XXX.T, y)
    return w


def init(file):
    father, son = [], []
    data = open(file, 'r').readlines()
    for d in data:
        d = d.split()
        if len(d):
            father.append(float(d[0]))
            son.append(float(d[1]))
    return [father, son]


if __name__ == '__main__':
    
    filename = 'fatherandson.txt'
    height = init(filename)
    
    X = np.array([height[0]]).T
    y = np.array(height[1])
    
    bias = np.ones((1, len(X)))
    X_bias = np.concatenate((X, bias.T), axis=1)
    
    W = fit(X_bias, y)
    
    print(W)
