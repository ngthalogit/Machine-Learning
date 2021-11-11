"""

auth: Nguyen Thanh Long
This program implements Naive Bayes Classifier with Multnominal Bayes from scratch

"""

# lib
import numpy as np

# Preprocessing data
DATA = [['hanoi pho chaolong hanoi'],
        ['hanoi buncha pho omai'],
        ['pho banhgio omai'],
        ['saigon hutiu banhbo pho']]

# labels
LABELS = ['B', 'B', 'B', 'N']

# test case
TEST_CASE = [['hanoi hanoi buncha hutiu'],
             ['pho hutiu banhbo']]

# get dictionary of dataset
paragraph = ""
for i in range(len(DATA)):
    paragraph += DATA[i][0]
    if i < len(DATA) - 1:
        paragraph += " "
DICT = [i for i in set(paragraph.split(" "))]

# length of list
d = len(DATA)
l = len(LABELS)
t = len(TEST_CASE)
N = len(DICT)


# formating input data
def data_process(DATA, DICT, N, d):
    data_arr = np.array([])
    for i in range(d):
        data_row = []
        splited = DATA[i][0].split(" ")
        data_row.append([splited.count(DICT[j]) for j in range(N)])
        data_arr = np.append(data_arr, data_row)
    return data_arr.reshape(-1, N)

# probability class c in input data
def p(c, labels):
    return np.count_nonzero(labels == c) / len(labels)


def pMultinomial(c, n, labels, X_train, alpha=1):
    total = np.sum(X_train[labels == c, :], axis=0)
    total = (total + alpha) / (np.sum(total, axis=0) + alpha * n)
    return total

# predict
def pred(X_train, y_train, X_test, n):
    rs = []
    for t in X_test:
        tmp_rs = []
        for l in set(y_train):
            pC = p(l, y_train)
            p_multinomial = pMultinomial(l, n, y_train, X_train)
            tmp_rs.append([pC * np.prod(p_multinomial[:] ** (t)), l])
        # rs.append(max(tmp_rs, key=lambda x: x[0]))
        rs.append(tmp_rs)
    return rs


def get_prob(rs):
    labels = np.array([rs[0][i][1] for i in range(len(rs[0]))])
    probab = np.array([])
    for r in rs:
        values = np.array([r[i][0] for i in range(len(r))])
        values = np.round(values * 100 / np.sum(values), 2)
        probab = np.append(probab, values)
    return probab.reshape(-1, len(labels)), labels


# init dataset
X_train = data_process(DATA, DICT, N, d)
y_train = np.array(LABELS)
X_test = data_process(TEST_CASE, DICT, N, t)
y_test = np.array(["B"])

rs = pred(X_train, y_train, X_test, N)

probab, labels = get_prob(rs)
