"""

auth: Nguyen Thanh Long
This program optimizes linear programming by simplex method

"""

import numpy as np

# func
def is_same_col(lcol, rcol):
    for b in np.array(lcol) == np.array(rcol):
        if b == False: return False
    return True


def is_in_array(col, narray):
    if narray.size != 0:
        n = narray.shape[1]
        for idx in range(n):
            val = narray[:, idx]
            if is_same_col(col, val): return True
    return False


def to_canonical(constrains):
    I = np.eye(constrains.shape[0])
    for idx in range(I.shape[1]):
        val = I[:, idx]
        if not is_in_array(val, constrains):
            constrains = np.concatenate((constrains, np.array([val]).T), axis=1)
    return constrains


def to_bigM_form(A, G, obj_func):
    G_can = np.copy(G)
    n = G.shape[1] - A.shape[1]
    if n > 0:
        A = np.concatenate((A, np.zeros((A.shape[0], n))), axis=1)
        G = np.concatenate((G, A), axis=0)
    G = to_canonical(G)
    m = G.shape[1] - G_can.shape[1]
    M = 10000000
    obj_func = np.concatenate((obj_func.reshape(1, -1), M * np.ones((1, m))), axis=1)
    return G, obj_func[0]


def constrains_modified(obj_func, constrains, h, A=None, b=None):
    if type(A) != type(None):
        G_can = to_canonical(constrains)
        G, obj_func = to_bigM_form(A, G_can, obj_func)
        h = np.concatenate((h.reshape(1, -1), b.reshape(1, -1)), axis=1)
        h = h[0]
    else:
        G = to_canonical(constrains)
    return obj_func, G, h


def init_table(f, G, h, n, m=0, bigM=False):
    heigh = 1 + n
    wide = 2 + heigh + len(f)
    table = np.zeros((heigh, wide))
    table[1:, 0] = [i for i in range(2 + len(f), 2 + len(f) + n)]
    table[0, 2: 2 + len(f)] = -f
    if m > 0:
        M = 10000000
        table[0, wide - 1 - m:-1] = M * np.ones((1, m))
    table[1:, 2: 2 + G.shape[1]] = G
    table[1:, -1] = h
    return table


def get_pivot_index(table):
    tmp = np.copy(table)
    idx = np.argmin(tmp[0, :])
    for i in range(1, tmp.shape[0]):
        tmp[i, idx] = tmp[i, -1] / tmp[i, idx] if tmp[i, idx] > 0 else 10000000000
    return np.argmin(tmp[1:, idx]) + 1, idx


def optimize(table):
    while True:
        r, c = get_pivot_index(table)
        pivot = table[r, c]
        table[r, 0] = c
        table[r, 2:] /= pivot
        for i in range(table.shape[0]):
            table[i, 2:] -= table[i, c] * table[r, 2:] if i != r else 0
        if np.all(table[0, 2:-1] >= 0):
            break
    rs = np.zeros((1, table.shape[1]))
    for i in range(1, table.shape[0]):
        rs[0, int(table[i, 0])] = table[i, -1]
    rs[0, -1] = table[0, -1]
    return rs


def solve(obj_func, constrains, h, A=None, b=None):
    f, G, h = constrains_modified(obj_func, constrains, h, A, b)
    m = len(f) - len(obj_func)
    n = G.shape[1] - constrains.shape[1]
    if n > 0:
        table = init_table(obj_func, G, h, n, m, bigM=True)
    else:
        table = init_table(obj_func, G, h, n)
    rs = optimize(table)
    return rs[0, 2:]


