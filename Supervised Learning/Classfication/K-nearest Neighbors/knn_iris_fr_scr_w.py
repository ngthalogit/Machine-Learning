"""
auth: Nguyen Thanh Long

This program implements the knn algorithm from scratch on iris datasets 

"""

# libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

# functions for distance computations
def dist_pp(z, x): # distance between point and point
    return np.sum((z - x) * (z - x))

def dist_ps_faster(z, X): # distance between point to set of points following the formula
   z2 = np.sum(z * z)
   X2 = np.sum(X * X, 1)
   return z2 + X2 - 2 * np.dot(X, z)

def dist_ss(Z, X): # distance between set to set
    Z2 = np.sum(Z * Z, 1)
    X2 = np.sum(X * X, 1)
    return  Z2.reshape(-1, 1) + X2.reshape(1, -1) - 2 * np.dot(Z, X.T)

# constances
SEED = 5
TEST_SIZE = 130
TRAIN_SIZE = 20
N_NEIGHBORS = 11
NORM = 2

# init datasets
np.random.seed(SEED)
iris_ds = datasets.load_iris()
iris_X = iris_ds.data
iris_y = iris_ds.target
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=TEST_SIZE)
labels = np.unique(y_train)

# distance between X_test and X_train
dis = dist_ss(X_test, X_train)

# add labels for sorting
res = []
for i in range(TEST_SIZE):
    row = []
    for j in range(TRAIN_SIZE):
        if dis[i][j] != 0:
            weight = 1/dis[i][j]
        else:
            weight = -1
        col = [dis[i][j], y_train[j], weight]
        row.append(col)
    res.append(row)

# sort each row by distance
for i in range(TEST_SIZE):
    res[i].sort(key=lambda x: x[0])

y_pred = []
for i in range(TEST_SIZE):
    w = [0 for w in range(len(labels))]
    for j in range(N_NEIGHBORS):
        if res[i][j][-1] == -1:
            y_pred.append(res[i][j][1])
            break
        else:
            w[res[i][j][1]] += res[i][j][-1]
    y_pred.append(w.index(max(w)))

# find accuracy
accuracy = 0
for i in range(TEST_SIZE):
    if y_pred[i] == y_test[i]:
        accuracy += (1/TEST_SIZE) * 100


print(accuracy)









