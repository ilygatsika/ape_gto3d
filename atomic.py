import numpy as np
import h5py
import matplotlib.pyplot as plt

# R_nl, (m = -l, .., l)
# n = l+1, .., nmax
# 

# Input file atomic
atom_file = 'dat/1e.chk'

f1 = h5py.File(atom_file)

# valeurs et vecteurs propres
lmax = 0
E = []
orbs = []
for l in range(lmax + 1):
    E.append(np.array(f1[f'E_{l}']))
    orbs.append(np.array(f1[f'orbs_{l}']))

# quadrature points
r = np.array(f1["r"]).flatten()

# quadrature weights
wr = np.array(f1["wr"]).flatten()

print(orbs[0].shape)
print(r.shape)

for n in range(1,lmax + 1):
    print(f'{-0.5/n**2}')


# liste qui contient tous les vecteurs de la même énergie
A = orbs[0] @ np.diag(np.multiply(np.square(r), wr)) @ orbs[0].T 

#print(A) 
print(np.all(np.isclose(A, np.eye(139))))

#B = orbs[1] @ np.diag(np.multiply(np.square(r), wr)) @ orbs[1].T 
#print(np.all(np.isclose(B, np.eye(139))))

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

