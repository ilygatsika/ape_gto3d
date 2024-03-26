import src.fem as fem
import src.read as read
import src.gto as gto
import src.partition as pou
from pyscf.solvent import ddcosmo
from pyscf.symm import sph
from pyscf import dft
import numpy as np
import matplotlib.pyplot as plt

"""
Green's function term
* dV_pot par atomic code
* paralleliser les boucles
* pour faire le résidu : ajouter le terme avec le alpha
* 1./4pi coefficient devant l'intégrale
"""

# Parameters for partition overlap 
amin = 0.5
amax = 0.8 # max 0.9
# Parameters for spectral decomposition
nmax = 4 # lmax <= 15 due to PySCF
shift = 4.0 # 3.80688477

# Input files
density_file = 'dat/density_small.hdf5'
helfem_res_file = 'dat/helfem_small.chk'

# Read data Diatomic
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2, T_fem, P = read.diatomic_density(density_file)

def inner_projection(u1, u2, dV=dV):
    return np.sum(u1 * u2 * dV)

# Coulomb potential integration
dV_pot = fem.build_dV_pot_atomic(helfem_grid, wquad)

# Transfrom HelFEM grid to cart
coords = fem.prolate_to_cart(Rh, helfem_grid)
ncoords = coords.shape[0]
print("coords shape", coords.shape)

# Function to integrate 
f = lambda x: np.exp(-(x[0]**2 + x[1]**2 + x[2]**2))

# Partition of unity
def partition(coords):

    nuc_1 = np.array([0,0,-Rh])
    nuc_2 = np.array([0,0,+Rh])
    x_1 = coords - nuc_1
    x_2 = coords - nuc_2
    vec_r_1 = np.array([np.linalg.norm(x) for x in x_1])
    vec_r_2 = np.array([np.linalg.norm(x) for x in x_2])
    part_1 = pou.partition_vec(vec_r_1, amin, amax)
    part_2 = pou.partition_vec(vec_r_2, amin, amax)
    part_3 = 1 - part_1 - part_2

    return part_3

part_3 = partition(coords)

print("partition of unity evaluated at grid ", part_3.shape)
#print(part_3)

#plt.plot(part_3)
#plt.show()

# DEBUG
"""
alpha = 0 #np.sqrt(shift)
f_y_1 = np.empty(ncoords)
f_y_2 = np.empty(ncoords)
for i in range(ncoords):
    
    y = coords[i]
    f_z_1 = np.empty(ncoords)
    f_z_2 = np.empty(ncoords)
    for j in range(ncoords):
        z = coords[j]
        # normalized Gaussian 1
        c = pow(1./np.pi, 3/2)
        f_z_1[j] = c * f(y + z)
        # normalized Gaussian 2
        #c = pow(alpha/np.pi, 3/2)
        #f_z_2[j] = 1#c * np.exp(-alpha*(z[0]**2 + z[1]**2 + z[2]**2))
        f_z_2[j] = 1./np.sqrt(z[0]**2 + z[1]**2 + z[2]**2)#c * np.exp(-alpha*(z[0]**2 + z[1]**2 + z[2]**2))

    # Evaluate 3D integral on z
    #f_y_1[i] = inner_projection(f_z_1, f_z_2, dV_pot)
    f_y_1[i] = inner_projection(f_z_1, f_z_2, dV)

    c = pow(1./np.pi, 3/2)
    f_y_2[i] = c * f(y)  

# 5.226823600578967
# 0.938670208283878

# Evaluate 3D integral on y
val = inner_projection(f_y_1, f_y_2, dV)

print(val)

a = 0.5
print(np.sqrt(4*a/np.pi))
"""

Efem, Efem_kin, Efem_nuc, Efem_nucr, T, S, H0 = read.diatomic_energy(helfem_res_file)
Efem += shift

# Define PySCF Mole object
basis = 'cc-pvqz'
mol, E_gto, C = gto.build_gto_sol(Rh, basis) 
ao_value = dft.numint.eval_ao(mol, coords)
u_gto = ao_value @ C
E_gto += shift
# H(-X) = E(-X) par convention on prend la positive
flag = False
if inner_projection(u_fem, u_gto) < 0 :
    C = - C
    u_gto = - u_gto
    flag = True

# vrai valeur
alpha = np.sqrt(shift)
f_y_1 = np.empty(ncoords)

# Integrant on y
Res = gto.residual(mol, coords, C, E_gto, Rh, Z1, Z2, flag)
part_3 = partition(coords)
f_y_2 = np.sqrt(part_3) * Res

for i in range(ncoords):

    y = coords[i]
    
    # Grid on z
    z_vec = coords + y

    # Integrant evaluated on z + y
    Res = gto.residual(mol, z_vec, C, E_gto, Rh, Z1, Z2, flag)
    part_3 = partition(z_vec)
    f_z_1 = np.sqrt(part_3) * Res
    f_z_2 = np.array([1./np.sqrt(z[0]**2 + z[1]**2 + z[2]**2) for z in z_vec])

    # Evaluate 3D integral on z
    f_y_1[i] = inner_projection(f_z_1, f_z_2, dV)

# Evaluate 3D integral on y
factor = 1./(4*np.pi) 
val = inner_projection(f_y_1, f_y_2, dV)

print(val)

