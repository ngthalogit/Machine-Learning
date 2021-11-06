"""

auth: Nguyen Thanh Long
This program uses linear regression model in sklearn to predict son's height.

"""

# import Library
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Read data from .txt file
"""
    File includes xxx rows and 2 columns:
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

    # model
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    # get weight
    W = [regr.coef_[0], regr.intercept_]

    # predict
    y_pre = regr.predict(X_test)

    # evaluate model

    mse = mean_squared_error(y_test, y_pre)  # Mean Square Error

    rmse = np.sqrt(mse)  # Root Mean Square Error
