"""

auth: Nguyen Thanh Long
This program use linear regression model in sklearn to predict son's height.

"""

#import Library
import numpy as np
from sklearn import linear_model

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
    
    X = np.array([height[0]]).T # father's height
    y = np.array(height[1]) # son's height
    
    # add bias
    bias = np.ones((1, len(X)))
    X_bias = np.concatenate((X, bias.T), axis=1)
    
    # model
    regr = linear_model.LinearRegression()
    regr.fit(X_bias, y)
    
    # get weight
    W = [regr.coef_[0], regr.intercept_]
    
    print(W)
