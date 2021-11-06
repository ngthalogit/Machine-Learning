"""

auth: Nguyen Thanh Long
This program implements linear regression algorithm from scratch to pridect house's price.

"""

# import Library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# calculate
def fit(X, y):
    XX = np.linalg.pinv(np.dot(X, X.T))
    XXX = np.dot(XX, X)
    w = np.dot(XXX.T, y)
    return w

# predict test set
def predict(test, w):
    return np.dot(test, w.T)

""" evaluate model """

# Mean Square Error
def mse(actual, predicted):
    n = len(actual)
    return float(np.linalg.norm(actual - predicted) ** 2) * (1 / n)

# Root Mean Square Error
def rmse(actual, predicted):
    return np.sqrt(mse(actual, predicted))

if __name__ == '__main__':
    filename = 'kc_house_data.csv'
    data = pd.read_csv(filename)[0:500]  # load data from .csv file, we just take the first 499 samples to calculate

    y = np.array(data['price'].values)  # house's price
    data = data.drop(['date', 'price', 'id'], axis=1)
    X = np.array(data.values)  # 19 features includes 'bedrooms', 'bathrooms',...

    # add bias
    bias = np.ones((1, len(X)))
    X_bias = np.concatenate((X, bias.T), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.2, random_state=10)  # split X, y into
    # train and test set

    W = fit(X_train, y_train).T

    y_pre = predict(X_test, W)

    # evaluate model
    mse_val = mse(y_test, y_pre)  # Mean Square Error

    rmse_val = rmse(y_test, y_pre)  # Root Mean Square Error

   

