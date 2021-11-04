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
T = [['hanoi hanoi buncha hutiu'],
     ['pho hutiu banhbo']]

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


def pMultinomial(c, n, labels, X_train, alpha=1):
    total = np.sum(X_train[labels == c, :], axis=0)
    total = (total + alpha) / (np.sum(total, axis=0) + alpha * n)
    return total


def fit(X_train, y_train, X_test, n):
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


# init dataset
X_train = data_process(D, DICT, N, d)
y_train = np.array(L)
X_test = data_process(T, DICT, N, len(T))
y_test = np.array(["B"])

rs = fit(X_train, y_train, X_test, N)

probab = np.array([])
for r in rs:
    values = np.array([r[i][0] for i in range(len(r))])
    values = np.round(values * 100 / np.sum(values), 2)
    probab = np.append(probab, values)
labels = np.array([r[i][1] for i in range(len(rs[0]))])
probab = probab.reshape(-1, len(T))

print(probab, end="\n")
print(labels)
