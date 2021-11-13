import numpy as np
from linear_programming import solve

"""
Objective function: (x, y) = argmax( 5x + 3y ) 
Constrains:  x  +  y  <= 10 
            2x  +  y  <= 16 
             x  + 4y  <= 32 
            x,y       >= 0
"""

if __name__ == '__main__':
    # init data
    f = np.array([5., 3.])
    G = np.array([[1., 1.],
                  [2., 1.],
                  [1., 4.]])
    h = np.array([10., 16., 32.])

    rs = solve(f, G, h)

    root = rs[:len(f)]

    minVal = rs[-1]

