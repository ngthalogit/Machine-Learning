"""
auth: Nguyen Thanh Long

This program implements the knn algorithm from scratch on iris datasets

"""

# libraries
from distance_compute import dist_ss
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np


# constances
SEED = 1
TEST_SIZE = 130
TRAIN_SIZE = 20
N_NEIGHBORS = 3
NORM = 2

# init datasets
np.random.seed(SEED)
iris_ds = datasets.load_iris()
iris_X = iris_ds.data
iris_y = iris_ds.target
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=TEST_SIZE)


# distance between X_test and X_train
dis = dist_ss(X_test, X_train)

# add labels for sorting
res = []
for i in range(TEST_SIZE):
    row = []
    for j in range(TRAIN_SIZE):
        col = [dis[i][j], y_train[j]]
        row.append(col)
    res.append(row)

# sort each row by distance
for i in range(TEST_SIZE):
    res[i].sort(key=lambda x: x[0])

# get the first kth labels after sorting by distance
neighbors_lables = []
for i in range(TEST_SIZE):
    temp = res[i][:N_NEIGHBORS][:]
    neighbors_lables.append([j[1] for j in temp])

# find labels
y_pred = []
y_pred.append([max(set(LIST), key=LIST.count) for LIST in neighbors_lables])

# find accuracy
accuracy = 0
for i in range(TEST_SIZE):
    if y_pred[0][i] == y_test[i]:
        accuracy += (1/TEST_SIZE) * 100


print(accuracy)







