import src.fem as fem
import src.utils as utils
import src.gto as gto
import src.partition as pou
import src.norm as norm
from pyscf import dft
import numpy as np
import pymp
import sys

"""
Main code for H-norm error estimation using practical and guaranteed estimators
Estimator is Theorem 3.7
"""

# Options
basis = str(sys.argv[1]) # GTO basis set
resfile = str(sys.argv[2]) # file to store results

# Parameters for partition overlap 
amin = 0.1 # if we put larger, such as 0.5, the constant C_P exploses
amax = 0.8 # max 0.9
shift = 4.0 # 3.80688477
shift_inf = 3
sigmas = (3, 3, shift_inf, shift)
# Parameters for spectral decomposition
lmax = 6 # lmax <= 15 due to PySCF
lebedev_order = 13

# Input files
density_file = 'dat/density.hdf5'
helfem_res_file = 'dat/helfem.chk'
atom_file = 'dat/1e_lmax20_Rmax1_4.chk'

# Read data Diatomic
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = utils.diatomic_density(density_file)

def inner_projection(u1, u2, dV=dV):
    return np.sum(u1 * u2 * dV)

# Transfrom HelFEM grid to cart
coords = fem.prolate_to_cart(Rh, helfem_grid)
ncoords = coords.shape[0]
print("coords shape", coords.shape)

# Reference FEM solution from HelFEM
Efem, E2, Efem_kin, Efem_nuc, Efem_nucr = utils.diatomic_energy(helfem_res_file)
Efem = Efem - Efem_nucr + shift
# shift
E2 += shift

# Approximate GTO solution from PySCF
mol, E_gto, C = gto.build_gto_sol(Rh, basis) 
ao_value = dft.numint.eval_ao(mol, coords)
u_gto = ao_value @ C
# Shift
E_gto += shift
# H(-X) = E(-X) par convention on prend la positive
flag = inner_projection(u_fem, u_gto)
if ( flag < 0 ):
    u_gto = - u_gto

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
alpha = np.sqrt(shift_inf)
kernel = lambda x: 1./(4*np.pi) * \
        np.exp(-alpha * (x[0]**2 + x[1]**2 + x[2]**2))/np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
# integrand
p_res = lambda xv: np.sqrt(pou.partition_compl(Rh, xv, amin, amax)) * \
        gto.residual(mol, xv, C, u_fem, E_gto, flag, Rh, Z1, Z2, shift)

estim_Delta = norm.green_inner(p_res, kernel, coords, dV)

print('estim_Delta=',estim_Delta)

"""
Atomic estimator
"""

# Read data atomic
E_atom, orbs_rad, r_rad, w_rad = utils.atomic_energy(atom_file, lmax)

# Partition of unity evaluated on radial part
g = np.sqrt(pou.partition_vec(r_rad, amin, amax))
# residual
f = lambda xv: gto.residual(mol, xv, C, u_fem, E_gto, flag, Rh, Z1, Z2, shift)

eigpairs = (E_atom, orbs_rad)
rad_grid = (r_rad, w_rad)
estim_atom = norm.atom_inner(f, g, eigpairs, rad_grid, lebedev_order, lmax, shift)
print('estim_atom=', estim_atom)

r1 = 2*estim_atom + estim_Delta

# Now multiply by constants according to Theorem 3
# C is cP
# C tilde is c1
# C hat is c2.7
final_estim = pow(cP * 1./c1 * r1 + Efem * cP**2 * 1./c2**2 * r1**2, 0.5)
print("Estimator of Theorem 3.7=", final_estim)

# True Hnorm error
# Laplacian term
u_Delta_gto = gto.build_Delta(mol, coords, C)
val_Delta = - 0.5 * inner_projection(u_fem, u_Delta_gto)
# Potential V (Coulomb)
dV_pot = fem.build_dV_pot(helfem_grid, Z1, Z2, wquad)
val_pot = - np.power(Rh,2) * inner_projection(u_fem, u_gto, dV=dV_pot)
val_ovlp = inner_projection(u_gto, u_fem)

# Hnorm 
err_H = np.sqrt(Efem + E_gto - 2*( - val_Delta + val_pot + shift * val_ovlp))

print("True error (H norm)", err_H)

# Store results to file
data = {}
data["err_H"] = err_H
data["estimator"] = final_estim
# parameters
data["amin"] = amin
data["amax"] = amax
data["shift"] = shift
# Estimator constants
data["cH"] = cH
data["c1"] = c1
data["c2"] = c2
data["cP"] = cP
# Estimators on domains
data["estim_atom"] = estim_atom
data["estim_Delta"] = estim_Delta
data["shift1"] = sigmas[0] 
data["shift2"] = sigmas[1] 
data["shift3"] = sigmas[2] 
data["lmax"] = lmax
data["lebedev_order"] = lebedev_order
data["density_file"] = density_file
data["helfem_res_file"] = helfem_res_file
data["atom_file"] = atom_file
key = basis
utils.store_to_file(resfile, key, data)


