"""

auth: Nguyen Thanh Long

"""

# import Library
import numpy as np


# calculate derivative function f(x) =  x ** 2  + 10 * sin(x)
def derivative(x):
    return 2 * x + 10 * np.cos(x)


# calculate cost of function f(x) = x ** 2 + 10 * sin(x)
def function_cost(x):
    return x ** 2 + 10 * np.sin(x)


# calculate gradient descent
def grad_descent_mmt(x, learning_rate, eps, gm):
    iter_count = 0
    v = 0  # initiate velocity
    v = gm * v + learning_rate * derivative(x)
    x = x - v
    val = derivative(x)
    while np.abs(val) > eps:
        v = gm * v + learning_rate * derivative(x)
        x = x - v
        val = derivative(x)
        iter_count += 1
    return x, iter_count


if __name__ == "__main__":
    lr = 0.1  # learning rate
    epsilon = 1e-3  # epsilon
    gamma = 0.9 # gamma
    x_0 = -5  # initial value

    root, iterate = grad_descent_mmt(x_0, lr, epsilon, gamma)

    minVal = function_cost(root)  # the smallest value
    
 

    

