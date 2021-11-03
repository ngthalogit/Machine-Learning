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


def lamdaC(c, n, labels, X_train, alpha=1):
    total = np.sum(X_train[labels == c, :], axis=0)
    total = (total + alpha) / (np.sum(total, axis=0) + n)
    return total


def fit(X_train, y_train, X_test, n):
    rs = []
    for t in X_test:
        tmp_rs = []
        for l in set(y_train):
            pC = p(l, y_train)
            ldC = lamdaC(l, n, y_train, X_train)
            tmp_rs.append([pC * np.prod(ldC[:] ** (t)), l])
        #rs.append(max(tmp_rs, key=lambda x: x[0]))
        rs.extend(tmp_rs)
    return rs


# init dataset
X_train = np.array(data_process(D, DICT, N, d))
y_train = np.array(L)
X_test = np.array(data_process(T, DICT, N, len(T)))
y_test = np.array(["B"])

rs = fit(X_train, y_train, X_test, N)

y_pred = max(rs, key=lambda x: x[0])[1]

print(y_pred)