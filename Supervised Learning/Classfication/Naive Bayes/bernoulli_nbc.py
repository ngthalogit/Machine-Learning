# lib
import numpy as np

# Preprocessing data
D = [['hanoi pho chaolong hanoi'],
     ['hanoi buncha pho omai'],
     ['pho banhgio omai'],
     ['saigon hutiu banhbo pho']]
d = len(D)
# labels
L = ['B', 'B', 'B', 'N']
# test case
T = [['hanoi hanoi buncha hutiu']]

paragraph = ""
for i in range(len(D)):
    paragraph += D[i][0]
    if i < len(D) - 1:
        paragraph += " "
DICT = [i for i in set(paragraph.split(" "))]
N = len(DICT)


# function
def data_process(DATA, DICT, N, d):
    data_arr = np.array([])
    for i in range(d):
        data_row = []
        splited = DATA[i][0].split(" ")
        data_row.append([splited.count(DICT[j]) for j in range(N)])
        data_arr = np.append(data_arr, data_row)
    return data_arr.reshape(-1, N)


def p(c, labels):
    return np.count_nonzero(labels == c) / len(labels)


def pBernoulli(c, n, labels, occur_train, alpha=1):
    # TODO: define the probability with Bernoulli theory
    pass


def fit(X_train, X_test, y_train, occur_train):
    # TODO: define
    pass




occur_train = data_process(D, DICT, N, d)
occur_test = data_process(T, DICT, N, len(T))

X_train = np.where(occur_train > 0, 1, occur_train)
y_train = np.array(L)
X_test = np.where(occur_test > 0, 1, occur_test)
y_test = np.array(["B"])
