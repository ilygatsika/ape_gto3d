import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.special import assoc_laguerre
from math import factorial as fact
import src.read as read

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



# Input file atomic
atom_file = 'dat/1e_lmax20_Rmax40.chk'

E, orbs, r, wr = read.atomic_energy(atom_file, lmax=6)

#print(orbs[0].shape)
#print(r.shape)

lmax = 5
for n in range(1,lmax + 1):
    print(f'{-0.5/n**2}')

print(E)

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
plt.plot(r, r**2*orbs[0][2]**2, label='3s')
plt.plot(r, r**2*orbs[0][3]**2, label='4s')
plt.xlim(0, 10)
plt.legend()
plt.savefig("img/s.png")
plt.close()

plt.plot(r, r**2*orbs[1][1]**2, label='2px')
plt.plot(r, r**2*orbs[1][2]**2, label='3px')
plt.plot(r, r**2*orbs[1][3]**2, label='4px')
plt.xlim(0, 20)
plt.legend()
plt.savefig("img/ps.png")
plt.close()

