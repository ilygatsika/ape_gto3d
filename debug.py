import sympy
import numpy as np
import src.read as read
import src.fem as fem
from src.partition import eval_supremum as test
from src.partition import partition_compl as test_2

"""
Problem avec FEM grid is the quality around zero
"""

amin, amax = 0.5, 0.8
sigmas = (3, 3, 3, 9)

density_file = 'dat/density.hdf5'
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = read.diatomic_density(density_file)
coords_init = fem.prolate_to_cart(Rh, helfem_grid)
print(coords_init.shape)
coords = np.zeros(coords_init.shape)
coords[:,2] = np.sort(coords_init[:,2])
#test_2(Rh, coords, a, b, plot=True)

X = np.linspace(-2*Rh, 2*Rh, 500)
coords = np.zeros((X.shape[0], 3))
coords[:,2] = X
#test_2(Rh, coords, a, b, plot=True)

#coords = fem.prolate_to_cart(Rh, helfem_grid)
test(coords, amin, amax, Rh, Z1, Z2, sigmas, plot=True)


exit()

test(a, b, Rh, Z1, Z2, sigmas)

exit()


r,a,b, = sympy.symbols('r a b')
f,g1,g2 = sympy.symbols('f g1 g2', cls=sympy.Function)

g1 = sympy.exp(-1/(b-r))
g2 = sympy.exp(-1/(r-a))
f = g1/(g1+g2)

# Laplacian
Delta_f = sympy.diff(f,r,2) + 2/r * sympy.diff(f,r)

# Gradient
nabla_f_r = sympy.diff(f,r)
nabla_f_theta = 0
nabla_f_phi = 0

Delta_f_fun = sympy.lambdify([a,b,r], Delta_f)
Delta_f_vfun = np.vectorize(Delta_f_fun)
print(Delta_f_fun(1,2,3))

delta = 0.05
vec_r = np.linspace(1.005,1.99,100)
val_1 = Delta_f_vfun(1,2,vec_r)
print(val_1)

import matplotlib.pyplot as plt

val = Delta_f_vfun(1,2,vec_r)
print(val)

plt.plot(vec_r, val)
#plt.plot(vec_r_1, val_1)
plt.show()
plt.close()



