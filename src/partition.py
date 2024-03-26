import numpy as np
from math import exp, log

"""
Routines for 2D partition function definition
- is radial
- is regular enough
- is division-by-zero aware
"""

def delta_value(a, b):
    """
    Numerical threshold for evaluating fun on limit points
    """
   
    # Use double precision epsilon
    eps = np.finfo(float).eps

    # Equation to solve for s 
    g = lambda s: 1/(-log(eps * exp(-1/(b-a-s))))

    # Fixed point iteration
    d = 0.1
    while True:
        d_new = g(d)
        if d == d_new:
            break
        d = d_new

    return d

def partition(x, a, b):
    """
    Partition of unity function p(x) for x>0
    with support on (0,b), decreasing on (a,b) and constant on (0,a)
    """

    # Numerical interval for avoiding division by zero
    delta = delta_value(a, b)

    if (x < a + delta):
        return 1
    elif (x > b - delta):
        return 0
    else:
        return 1/(1+exp(-1/(x-a))/exp(-1/(b-x)))

def partition_vec(x, a, b):
    """
    Evaluate partition function on a vector
    """

    delta = delta_value(a,b)
    #print(f'{delta=}')
    idx_1 = np.where(x < a + delta)[0]
    idx_0 = np.where(x > b - delta)[0]
    n = np.size(x)
    mask = np.ones(n, dtype=bool)
    mask[idx_1] = False
    mask[idx_0] = False
    y = np.ones(n, dtype=float)
    y[idx_0] = 0
    y[mask] = [partition(xpt, a, b) for xpt in x[mask]]

    return y

# Debug
"""
a, b = 1, 2
print("Partition at x=a is %.16f" %partition(a, a, b))
print("Partition at x=b is %.16f" %partition(b, a, b))

import matplotlib.pyplot as plt
x = np.linspace(a, b, 10000)
plt.plot(x, partition_vec(x, a, b))
plt.show()
plt.close()
"""

