"""

auth: Nguyen Thanh Long

This is program implement linear regression algorithm from scratch to predict height'son. 

"""

# import Library
import numpy as np

#calculate 
def fit(X_bias, y):
    XX = np.linalg.pinv(np.dot(X_bias, X_bias.T))
    XXX = np.dot(XX, X_bias)
    w = np.dot(XXX.T, y)
    return w

# Read data from .txt file
"""
    File includes 1087 rows and 2 columns:
        The first columns is height's father
        The second columns is height's son 
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
    
    X = np.array([height[0]]).T
    y = np.array(height[1])
    
    # add bias 
    bias = np.ones((1, len(X)))
    X_bias = np.concatenate((X, bias.T), axis=1)
    
    W = fit(X_bias, y)
    
    print(W) 
