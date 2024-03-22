import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.special import assoc_laguerre
from scipy.special.constants import physical_constants
from math import factorial as fact


# R_nl, (m = -l, .., l)
# n = l+1, .., nmax
# 

"""
Eigenstates for Hydrogen atom:

    Every state is denoted by nl,
    where n:energy level or principal quantum number 
          l:orbital quantum number (l<n), 
    where n=1,2,3,... and l=0,1,2,...,n-1. Then we also have m=-l,...,+l that
    counts for degeneracy

    n=1, l=0 -> 1s
    n=2, l=0 -> 2s
         l=1 -> 2p
    n=3, l=0 -> 3s
         l=1 -> 3p
         l=2 -> 3d
"""

# Bohr radius
a0, _, _ = physical_constants["Bohr radius"]

def radial(n,l,r):
    """
    Return radial part of Hydrogen wave function

    McQuarrie, page 223
    """
    # normalisation factor
    a = fact(n-l-1)
    b = 2*n*pow(fact(n+l), 3)
    c = pow(2/(n*a0), l + 3/2)
    nrml = -pow(a/b, 0.5) * c

    # exponential
    expo = pow(r,l) * np.exp(-r/(n*a0))

    # associated Laguerre polynomial L_n^k
    poly = assoc_laguerre(2/(n*a0), n+1, 2*l+1)

    return (nrml * expo * poly)


# Input file atomic
atom_file = 'dat/1e_lmax6.chk'

f1 = h5py.File(atom_file)

# valeurs et vecteurs propres, partie angulaire
lmax = 5
E = []
orbs = []
for l in range(lmax + 1):
    E.append(np.array(f1[f'E_{l}']))
    orbs.append(np.array(f1[f'orbs_{l}']))

# Radial part evaluated on quadrature points
r = np.array(f1["r"]).flatten()

# quadrature weights
wr = np.array(f1["wr"]).flatten()

#print(orbs[0].shape)
#print(r.shape)

for n in range(1,lmax + 1):
    print(f'{-0.5/n**2}')

print(E)
exit()

# liste qui contient tous les vecteurs de la même énergie
A = orbs[0] @ np.diag(np.multiply(np.square(r), wr)) @ orbs[0].T
n = A.shape[0]

print(np.all(np.isclose(A, np.eye(n))))

B = orbs[1] @ np.diag(np.multiply(np.square(r), wr)) @ orbs[1].T 
print(np.all(np.isclose(B, np.eye(n))))

print(orbs[0].shape)
#exit()
"""
def inner_projection(u1, u2, dV=dV):

    return np.sum(u1 * u2 * dV)
"""

#print(orbs[2].shape)

plt.title("orbital")
plt.plot(r, orbs[0][0],label='1s')
plt.plot(r, orbs[0][1], label='2s')
#plt.xlim(0, 10)
plt.legend()
plt.show()

plt.plot(r, r**2*orbs[0][0]**2,label='1s')
plt.plot(r, r**2*orbs[0][1]**2, label='2s')
#plt.xlim(0, 10)
plt.legend()
plt.show()

