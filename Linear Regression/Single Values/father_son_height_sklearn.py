"""

auth: Nguyen Thanh Long

"""
import numpy as np
from sklearn import linear_model


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
    filename = '/home/ntl0601/Downloads/fatherandson.txt'
    height = init(filename)
    
    X = np.array([height[0]]).T
    y = np.array(height[1])
    
    bias = np.ones((1, len(X)))
    X_bias = np.concatenate((X, bias.T), axis=1)
    
    regr = linear_model.LinearRegression()
    regr.fit(X_bias, y)
    
    w = [regr.coef_[0], regr.intercept_]
    print(w)



