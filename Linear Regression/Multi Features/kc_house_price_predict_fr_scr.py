"""

auth: Nguyen Thanh Long

"""
import numpy as np
import pandas as pd


def fit(X, y):
    XX = np.linalg.pinv(np.dot(X, X.T))
    XXX = np.dot(XX, X)
    w = np.dot(XXX.T, y)
    return w


if __name__ == '__main__':

    filename = 'kc_house_data.csv'
    data = pd.read_csv(filename)[0:500]

    y = np.array(data['price'].values)
    data = data.drop(['date', 'price', 'id'], axis=1)
    X = np.array(data.values)

    bias = np.ones((1, len(X)))
    X_bias = np.concatenate((X, bias.T), axis=1)
    
    W = fit(X_bias, y)

    print(W)