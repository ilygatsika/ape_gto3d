import sympy
import numpy as np
from src.partition import eval_supremum as test
from src.partition import partition_compl as test_2

a, b = 0.5, 0.8
Rh = 0.7
Z1, Z2 = 1, 1
sigmas = (3, 3, 3, 9)

vec_r = np.linspace(-5*Rh, 5*Rh, 500)
test_2(Rh, vec_r, a, b, plot=True)
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



