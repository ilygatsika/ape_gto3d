import src.fem as fem
import src.read as read
import src.gto as gto
import src.partition as pou
import src.norm as norm
from pyscf import dft
import numpy as np
import matplotlib.pyplot as plt
import pymp

"""
Main code for H-norm error estimation using practical and guaranteed estimators
Estimator is Theorem 3.7
"""

# Parameters for partition overlap 
amin = 0.1 # if we put larger, such as 0.5, the constant C_P exploses
amax = 0.8 # max 0.9
sigmas = (3, 3, 3, 9)
# Parameters for spectral decomposition
lmax = 6 # lmax <= 15 due to PySCF
lebedev_order = 13
shift = 4.0 # 3.80688477
basis = 'aug-cc-pvtz' # GTO basis set

# Input files
density_file = 'dat/density_small.hdf5'
helfem_res_file = 'dat/helfem_small.chk'
atom_file = 'dat/1e_lmax20_Rmax1_4.chk'

# Read data Diatomic
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = read.diatomic_density(density_file)

def inner_projection(u1, u2, dV=dV):
    return np.sum(u1 * u2 * dV)

# Transfrom HelFEM grid to cart
coords = fem.prolate_to_cart(Rh, helfem_grid)
ncoords = coords.shape[0]
print("coords shape", coords.shape)

# Reference FEM solution from HelFEM
Efem, E2, Efem_kin, Efem_nuc, Efem_nucr = read.diatomic_energy(helfem_res_file)
Efem -= Efem_nucr
# shift
Efem += shift
E2 += shift

print(Efem)

# Approximate GTO solution from PySCF
mol, E_gto, C = gto.build_gto_sol(Rh, basis) 
ao_value = dft.numint.eval_ao(mol, coords)
u_gto = ao_value @ C
# Shift
E_gto += shift
# H(-X) = E(-X) par convention on prend la positive
flag = ( inner_projection(u_fem, u_gto) < 0 )

# Constant of Assumption 3
cH = 1./Efem
# Gap constants for the first eigenvalue
c1 = (1 - E_gto / E2)**2 # equation 3.3, C_tilde
c2 = (1 - E_gto / E2)**2 * E2 # equation 3.4, C_hat

print('cH=',cH)
print('c1=',c1)
print('c2=',c2)

# Constant associated to partition (equation 1.3)
val_sup = pou.eval_supremum(amin, amax, Rh, Z1, Z2, sigmas)
cP = 1 + cH**2 * val_sup
print('cP=', cP)

# error should be 1.28e-01
print(basis, "eigenvalue", Efem, E_gto, "error", abs(Efem - E_gto))


"""
Laplacian estimator
takes some time
"""


# Green's function of the screened Laplacian operator
alpha = np.sqrt(shift)
kernel = lambda x: 1./(4*np.pi) * \
        np.exp(-alpha * (x[0]**2 + x[1]**2 + x[2]**2))/np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
# integrand
p_res = lambda xv: np.sqrt(pou.partition_compl(Rh, xv, amin, amax)) * \
        gto.residual(mol, xv, C, E_gto, Rh, Z1, Z2, flag, shift)

estim_Delta = norm.green_inner(p_res, kernel, coords, dV)


#estim_Delta = 0.21493942891276085 

print('estim_Delta=',estim_Delta)

"""
Atomic estimator
"""

# Read data atomic
E_atom, orbs_rad, r_rad, w_rad = read.atomic_energy(atom_file, lmax)

# Partition of unity evaluated on radial part
g = np.sqrt(pou.partition_vec(r_rad, amin, amax))
# residual
f = lambda xv: gto.residual(mol, xv, C, E_gto, Rh, Z1, Z2, flag, shift)

eigpairs = (E_atom, orbs_rad)
rad_grid = (r_rad, w_rad)
estim_atom = norm.atom_inner(f, g, eigpairs, rad_grid, lebedev_order, lmax, shift)

r1 = 2*estim_atom + estim_Delta

# Now multiply by constants according to Theorem 3
# C is cP
# C tilde is c1
# C hat is c2.7
final_estim = pow(cP * 1./c1 * r1 + Efem * cP**2 * 1./c2**2 * r1**2, 0.5)
print("Estimator of Theorem 3.7=", final_estim)

# True Hnorm error
# Laplacian term
lapl_ao = dft.numint.eval_ao(mol, coords, deriv=2)
u_Delta_gto = (lapl_ao[4] + lapl_ao[7] + lapl_ao[9]) @ C
val_Delta = inner_projection(u_fem, u_Delta_gto)
# Potential V (Coulomb)
dV_pot = fem.build_dV_pot(helfem_grid, Z1, Z2, wquad)
val_pot = - np.power(Rh,2) * inner_projection(u_fem, u_gto, dV=dV_pot)
# Convention to take positive
if flag : 
    u_Delta_gto = - u_gto 
    u_gto = - u_gto 

# Hnorm 
err_H = np.sqrt(Efem + E_gto - 2*( - 0.5 * val_Delta - val_pot + shift))
#err_H = norm.H_norm()

print("True error (H norm)", err_H)


