import numpy as np
from math import exp, log
import matplotlib.pyplot as plt
import sympy

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
    with support on (0,b), decreasing on (a,b) and constant (=1) on (0,a)
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

def partition_compl(Rh, vec_r, amin, amax, plot=False):
    """
    Partition of unity of the complement domain of two atoms

    Atoms positioned at -Rh and +Rh on z-axis
    """
    
    #vec_r = np.linalg.norm(coords, axis=1)
    p1 = partition_vec(vec_r, amin + Rh, amax + Rh)
    p2 = 1 - partition_vec(vec_r, amax + Rh, amin + Rh)
    p3 = 1 - p1 - p2
    
    if (plot):
        plt.plot(vec_r, p1, label="p1")
        plt.plot(vec_r, p2, label="p2")
        plt.plot(vec_r, p3, label="p3")
        plt.legend()
        plt.show()
        plt.close()

    return p3

def build_deriv_partition():
    """
    Return Laplacian and gradient of partition of unity 
    as a function of one variable and two parameters
    """

    r,a,b, = sympy.symbols('r a b')
    f,g1,g2 = sympy.symbols('f g1 g2', cls=sympy.Function)

    # Partition of unity on (a,b)
    g1 = sympy.exp(-1/(b-r))
    g2 = sympy.exp(-1/(r-a))
    f = g1/(g1+g2)

    # Laplacian
    Delta_f = sympy.diff(f,r,2) + 2/r * sympy.diff(f,r)

    # Gradient (is zero for theta and phi)
    nabla_f = sympy.diff(f,r)

    # Turn into functions with input array
    Delta_f_fun = sympy.lambdify([r, a, b], Delta_f)
    Delta_f_vfun = np.vectorize(Delta_f_fun)

    nabla_f_fun = sympy.lambdify([r, a, b], nabla_f)
    nabla_f_vfun = np.vectorize(nabla_f_fun)

    return (Delta_f_vfun, nabla_f_vfun)

def deriv_partition_vec(vfun, r, a, b):
    """
    Evaluate derivative partition function on a vector
    derivatives are zero outside [a,b]
    """

    delta = 5e-2 
    idx_0_l = np.where( r < a + delta )[0]
    idx_0_r = np.where( r > b - delta )[0]
    n = np.size(r)
    mask = np.ones(n, dtype=bool)
    mask[idx_0_l] = False
    mask[idx_0_r] = False
    y = np.zeros(n, dtype=float)
    y[mask] = [vfun(xpt, a, b) for xpt in r[mask]]

    return y

def eval_supremum(a, b, Rh, Z1, Z2, sigmas):
    """
    Evaluate supremum in constant (3.1) of paper
    Attention a + delta < r < b - delta
    """
    
    # Shift
    sigma1, sigma2, sigma3, sigma = sigmas

    # Build Laplacian and gradient of partition
    Delta, nabla = build_deriv_partition()

    npts = 500
    vec_r = np.linspace(-2*Rh, 2*Rh, npts)

    # Term centered on atom 1 at position -Rh
    V1 = np.vectorize(lambda r: -Z1/r)(vec_r + Rh)
    Delta_1 = deriv_partition_vec(Delta, vec_r + Rh, a - Rh, b - Rh)
    nabla_1 = deriv_partition_vec(nabla, vec_r + Rh, a - Rh, b - Rh)
    p1 = partition_vec(vec_r, - a - Rh, b + Rh)
    #tot_1 = - 0.5 * Delta_1 + nabla_1**2/(4*p1) + V1*(p1 - 1) + (sigma1 - sigma)*p1

    # Term centered on atom 2 at position +Rh
    V2 = np.vectorize(lambda r: -Z1/r)(vec_r - Rh)
    Delta_2 = deriv_partition_vec(Delta, vec_r - Rh, a + Rh, b + Rh)
    nabla_2 = deriv_partition_vec(nabla, vec_r - Rh, a + Rh, b + Rh)
    p2 = 1 - partition_vec(vec_r, - b - Rh, a + Rh)
    #tot_2 = - 0.5 * Delta_2 + nabla_2**2/(4*p2) + V2*(p2 - 1) + (sigma2 - sigma)*p2

    import matplotlib.pyplot as plt
    plt.plot(vec_r, V1)
    plt.plot(vec_r, p1)
    plt.plot(vec_r, V2)
    plt.plot(vec_r, p2)
    plt.show()
    exit()

    # Term centered on complement
    Delta_3 = - Delta_1 - Delta_2
    nabla_3 = - nabla_1 - nabla_2
    p3 = 1 - p1 - p2
    tot_3 = - 0.5 * Delta_3 + nabla_3**2/(4*p3) + (sigma3 - sigma)*p3

    # Take positive part of sum
    tot = tot_1 + tot_2 + tot_3
    print(tot)
    tot = tot[tot > 0]

    # Supremum
    max_val = np.amax(tot)

    return max_val


# Debug
"""
a, b = 1, 2
print("Partition at x=a is %.16f" %partition(a, a, b))
print("Partition at x=b is %.16f" %partition(b, a, b))

x = np.linspace(a, b, 10000)
plt.plot(x, partition_vec(x, a, b))
plt.show()
plt.close()
"""

