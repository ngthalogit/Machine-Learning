"""

auth: Nguyen Thanh Long
This program implements Nesterov accelerated gradient

"""

# import Library
import numpy as np


# calculate cost of function x ** 2 + 10 * sin(x)
def function_cost(x):
    return x ** 2 + 10 * np.sin(x)


# calculate derivative of function x ** 2 + 10 * sin(x)
def derivative(x):
    return 2 * x + 10 * np.cos(x)


# calculate
def nesterov_gradient_descent(x, eps, gm, lr):
    v = 0
    v = v * gm + lr * derivative(x)
    x = x - v
    iter_count = 0
    val = derivative(x)
    while np.abs(val) > eps:
        v = v * gm + lr * derivative(x - gm * v)
        x = x - v
        val = derivative(x)
        iter_count += 1
    return [x, iter_count]


# main
if __name__ == "__main__":
    x_0 = -5  # initiate value
    epsilon = 1e-3  # epsilon
    gamma = 0.9  # gamma
    n = 0.1  # learning rate

    root, iterate = nesterov_gradient_descent(x_0, epsilon, gamma, n)

    minVal = function_cost(root)


