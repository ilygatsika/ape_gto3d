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

def partition(x, a, b, delta):
    """
    Partition of unity function p(x) for x>0
    with support on (0,b), decreasing on (a,b) and constant (=1) on (0,a)
    """

    # Numerical interval for avoiding division by zero
    if (x < a + delta):
        return 1
    elif (x > b - delta):
        return 0
    else:
        return 1/(1+exp(-1/(x-a))/exp(-1/(b-x)))

def partition_vec(x, a, b, delta):
    """
    Evaluate partition function on a vector x of radial parts
    """

    #print(f'{delta=}')
    idx_1 = np.where(x < a + delta)[0]
    idx_0 = np.where(x > b - delta)[0]
    n = np.size(x)
    mask = np.ones(n, dtype=bool)
    mask[idx_1] = False
    mask[idx_0] = False
    y = np.ones(n, dtype=float)
    y[idx_0] = 0
    y[mask] = [partition(xpt, a, b, delta) for xpt in x[mask]]

    return y

def partition_compl(Rh, coords, amin, amax, delta, plot=False):
    """
    Partition of unity of the complement domain of two atoms
    evaluated on 3D grid

    Atoms positioned at -Rh and +Rh on z-axis
    """
    
    nuc = np.array([0,0,-Rh])
    rad_1 = np.linalg.norm(coords - nuc, axis=1)
    rad_2 = np.linalg.norm(coords + nuc, axis=1)
    p1 = partition_vec(rad_1, amin, amax, delta)
    p2 = partition_vec(rad_2, amin, amax, delta)
    p3 = 1 - p1 - p2
    
    if (plot):
        # z-section
        zvec = coords[:,2]
        plt.plot(zvec, p1, label="p1")
        plt.plot(zvec, p2, label="p2")
        plt.plot(zvec, p3, label="p3")
        plt.xlabel("z coord")
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

    # Turn into functions of 3 parameters
    Delta_f_fun = sympy.lambdify([r, a, b], Delta_f)
    nabla_f_fun = sympy.lambdify([r, a, b], nabla_f)

    return (Delta_f_fun, nabla_f_fun)

def eval_supremum(amin, amax, Rh, Z1, Z2, sigmas, delta, plot=False):
    """
    Evaluate supremum in constant (3.1) of paper
    over coords 3D Cartesian coordinates that is box around nuclei
    Attention amin + delta < |coord| < amax - delta
    """
    
    # Shift
    sigma1, sigma2, sigma3, sigma = sigmas

    # Build Laplacian and gradient of partition
    Delta, nabla = build_deriv_partition()

    # Coulomb potentials centered at zero
    V1 = lambda r: -Z1/r
    V2 = lambda r: -Z2/r
    
    # Distance of grid points from nuclei
    nuc = np.array([0,0,-Rh])
    # minimize over the z-axis
    X = np.linspace(-2*Rh, 2*Rh, 500)
    # minimize over the cube
    #X = np.linspace(-2*Rh, 2*Rh, 20)
    #vX, vY, vZ = np.meshgrid(X, X, X)
    #coords = np.vstack((vX.flatten(), vY.flatten(), vZ.flatten())).T
    coords = np.zeros((X.shape[0],3))
    coords[:,2] = X
    rad_1 = np.linalg.norm(coords - nuc, axis=1)
    rad_2 = np.linalg.norm(coords + nuc, axis=1)
    in_slice_1 = (amin + delta < rad_1) & (rad_1 < amax - delta)
    in_slice_2 = (amin + delta < rad_2) & (rad_2 < amax - delta)
     
    # For every 3D point evaluate function to minimize
    npts = rad_1.shape[0]
    vals = np.empty(npts, dtype=float)
    for i in range(npts):

        # get distance from atoms
        r1 = rad_1[i]
        r2 = rad_2[i]

        val1, val2, val3 = 0, 0, 0

        # Evaluate function on this coord 
        # if at slice 1 and not at slice 2
        if (in_slice_1[i] and (not in_slice_2[i])): 

            # Term on 1
            D1 = Delta(r1, amin, amax)
            g1 = nabla(r1, amin, amax)
            p1 = partition(r1, amin, amax, delta)
            val1 = - 0.25 * D1 + g1**2/(8*p1) + V1(r1) + (sigma1 - sigma)*p1
            
            # Complement term only depends on 1
            D3 = - D1
            g3 = - g1
            p3 = 1 - p1
            val3 = - 0.25 * D3 + (g3)**2/(8*p3) + (sigma3 - sigma) * p3

        # if at slice 2 and not at slice 1
        elif ((not in_slice_1[i]) and in_slice_2[i]): 

            # Term on 2
            D2 = Delta(r2, amin, amax)
            g2 = nabla(r2, amin, amax)
            p2 = partition(r2, amin, amax, delta)
            val2 = - 0.25 * D2 + g2**2/(8*p2) + V2(r2) + (sigma2 - sigma)*p2
            
            # Complement term only depends on 2
            D3 = - D2
            g3 = - g2
            p3 = 1 - p2
            val3 = - 0.25 * D3 + (g3)**2/(8*p3) + (sigma3 - sigma) * p3

        # if at slice 1 and 2
        elif (in_slice_1[i] and in_slice_2[i]):

            # Term on 1 
            D1 = Delta(r1, amin, amax)
            g1 = nabla(r1, amin, amax)
            p1 = partition(r1, amin, amax, delta)
            val1 = - 0.25 * D1 + g1**2/(8*p1) + V1(r1) + (sigma1 - sigma)*p1

            # Term on 2
            D2 = Delta(r2, amin, amax)
            g2 = nabla(r2, amin, amax)
            p2 = partition(r2, amin, amax, delta)
            val2 = - 0.25 * D2 + g2**2/(8*p2) + V2(r2) + (sigma2 - sigma)*p2 

            # Complement term only depends on 2
            D3 = - D1 - D2
            g3 = - g1 - g2
            p3 = 1 - p1 - p2
            val3 = - 0.25 * D3 + (g3)**2/(8*p3) + (sigma3 - sigma) * p3

        # Store value (positive part)
        vals[i] = max(val1 + val2 + val3, 0)

    if (plot):
        # z-section
        zvec = coords[:,2]
        plt.plot(zvec, vals)
        plt.xlabel("z coord")
        plt.show()
        plt.close()

    # Supremum
    max_val = np.amax(vals)

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

