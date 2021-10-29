"""

auth: Nguyen Thanh Long
This program implements gradient descent from scratch

"""

# import Library
import numpy as np


# calculate derivative function f(x) =  x * 2  + 5 * cos(x)
def derivative(x):
    return 2 * x + 5 * np.cos(x)


# calculate cost of function f(x) = x ** 2 + 5 * sin(x)
def function_cost(x):
    return x ** 2 + 5 * np.sin(x)


# calculate gradient descent
def grad_descent(x, learning_rate, eps):
    iter_count = 0
    x = x - learning_rate * derivative(x)
    val = derivative(x)
    while np.abs(val) > eps:
        x = x - learning_rate * derivative(x)
        val = derivative(x)
        iter_count += 1
    return x, iter_count


if __name__ == "__main__":
    lr = 0.1  # learning rate
    epsilon = 1e-3  # epsilon
    x_0 = -5  # initial value

    root, iterate = grad_descent(x_0, lr, epsilon)

    minVal = function_cost(root)  # the smallest values of function

