import numpy as np
from linear_programming import solve

"""
Objective function: (x, y, z) = argmin( -3x + y -2z ) => (x, y, z) = argmax( 3x - y + 2z ) 
Constrains: 2x +  4y - z  <= 10 
            3x +   y + z  >= 4  
             x +  4y + z  <= 32 
             x -   y + z   = 2
            x,y      >= 0
"""

if __name__ == '__main__':
    # init data
    f = np.array([3., -1., 2.])
    G = np.array([[2., 4., -3.],
                  [-3., -1., -1.]])
    h = np.array([10., -4.])
    A = np.array([[1., -1., 3.]])
    b = np.array([2.])

    rs = solve(f, G, h, A, b)

    root = rs[:len(f)]

    minVal = -rs[-1]
