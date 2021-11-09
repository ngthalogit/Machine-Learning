"""

auth: Nguyen Thanh Long
This program optimizes linear programming by simplex method

"""

import numpy as np

"""

Objective function: (x, y) = argmax( 5x + 3y ) => (x, y) = argmin( -5x - 3y )
Constrains: x  + y  <= 10 
            2x + y  <= 16 
            x  + 4y <= 32 
            x,y     >= 0
"""

# init data
obj_func = np.array([5, 3], dtype=float)  # objective function
left = np.array([[1, 1, 0, 0],
                 [2, 1, 1, 0],
                 [1, 4, 0, 1]], dtype=float)
right = np.array([10, 16, 32], dtype=float)
I = np.eye(left.shape[0], dtype=float)


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


for idx in range(I.shape[1]):
    val = I[:, idx]
    if not is_in_array(val, left):
        left = np.concatenate((left, np.array([val]).T), axis=1)


