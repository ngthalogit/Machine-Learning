"""

auth: Nguyen Thanh Long

"""
import numpy as np
import pandas as pd
from sklearn import linear_model

if __name__ == '__main__':

    filename = 'kc_house_data.csv'
    data = pd.read_csv(filename)[0:500]

    y = np.array(data['price'].values)
    data = data.drop(['date', 'price', 'id'], axis=1)
    X = np.array(data.values)

    bias = np.ones((1, len(X)))
    X_bias = np.concatenate((X, bias.T), axis=1)

    regr = linear_model.LinearRegression()
    regr.fit(X_bias, y)
    
    W = [regr.coef_[:-1], regr.intercept_]
    
    print(W)
