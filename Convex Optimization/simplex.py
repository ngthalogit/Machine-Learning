"""

auth: Nguyen Thanh Long
This program optimizes linear programming by simplex method

"""

import numpy as np

"""

Objective function: (x, y) = argmax( 5x + 3y ) 
Constrains: x  + y  <= 10 
            2x + y  <= 16 
            x  + 4y <= 32 
            x,y     >= 0
"""


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


def init_table(f, G, h, n):
    heigh = 1 + n
    wide = 2 + heigh + len(f)
    table = np.zeros((heigh, wide))
    table[1:, 0] = [i for i in range(2 + len(f), 2 + len(f) + n)]
    table[0, 2: 2 + len(f)] = -f
    table[1:, 2: 2 + G.shape[1]] = G
    table[1:,-1] = h
    return table


if __name__ == '__main__':
    # init data
    obj_func = np.array([5, 3], dtype=float)
    constrains = np.array([[1, 1],
                           [2, 1],
                           [1, 4]], dtype=float)
    h = np.array([10, 16, 32], dtype=float)
    I = np.eye(constrains.shape[0], dtype=float)
    G = to_canonical(constrains)
    n = G.shape[1] - constrains.shape[1]
    table = init_table(obj_func, G, h, n)
    print(table)
