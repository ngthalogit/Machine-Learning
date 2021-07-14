"""

auth: Nguyen Thanh Long

"""


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    filename = 'kc_house_data.csv'
    data = pd.read_csv(filename)[0:500]

    y = np.array(data['price'].values)
    data = data.drop(['date', 'price', 'id'], axis=1)
    X = np.array(data.values)

    bias = np.ones((1, len(X)))
    X_bias = np.concatenate((X, bias.T), axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.2, random_state=10)  # split X, y into
    # train and test set

    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    # get weight
    W = [regr.coef_[:-1], regr.intercept_]

    # predict
    y_pre = regr.predict(X_test)

    # evaluate model

    mse = mean_squared_error(y_test, y_pre)  # Mean Square Error

    rmse = np.sqrt(mse)  # Root Mean Square Error


