"""
auth: Nguyen Thanh Long
This program implement linear regression algorithm from scratch to predict son's height.
"""

# import Library
import numpy as np
from sklearn.model_selection import train_test_split


# calculate
def fit(X_bias, y):
    XX = np.linalg.pinv(np.dot(X_bias, X_bias.T))
    XXX = np.dot(XX, X_bias)
    w = np.dot(XXX.T, y)
    return w


# Read data from .txt file
"""
    File includes 1087 rows and 2 columns:
        The first columns is father's height.
        The second columns is son's height.
"""


def init(file):
    father, son = [], []
    data = open(file, 'r').readlines()
    for d in data:
        d = d.split()
        if len(d):
            father.append(float(d[0]))
            son.append(float(d[1]))
    return [father, son]


# predict test set
def predict(test, w):
    return np.dot(test, w.T)


""" evaluate model """


# Mean Square Error
def mse(actual, predicted):
    n = len(actual)
    return float(np.linalg.norm(actual - predicted) ** 2) * (1/n)


# Root Mean Square Error
def rmse(actual, predicted):
    return np.sqrt(mse(actual, predicted))


if __name__ == '__main__':
    filename = 'fatherandson.txt'
    height = init(filename)

    X = np.array([height[0]]).T  # father's height
    y = np.array(height[1])  # son's height

    # add bias
    bias = np.ones((1, len(X)))
    X_bias = np.concatenate((X, bias.T), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.2, random_state=10)  # split X, y into
    # train and test set

    W = fit(X_train, y_train).T

    y_pre = predict(X_test, W)

    # evaluate model 
    mse_val = mse(y_test, y_pre) # Mean Square Error
    
    rmse_val = rmse(y_test, y_pre) # Root Mean Square Error
    
