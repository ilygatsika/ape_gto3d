import src.fem as fem
import src.utils as utils
import src.gto as gto
import src.partition as pou
import src.norm as norm
from numpy.linalg import norm as norm2
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

"""
Main code for H-norm error estimation using practical and guaranteed estimators
Estimator is Theorem 3.7

This program outputs figures.
"""
# Available basis sets
basis_list = ['ccpvdz', 'ccpvtz', 'ccpvqz', 'ccpv5z']

# Parameters for partition overlap 
amin = 0.1 # if we put larger, such as 0.5, the constant C_P exploses
amax = 0.8 # max 0.9
shift = 4.0 # 3.80688477
shift_inf = 3
sigmas = (3, 3, shift_inf, shift)
# Parameters for spectral decomposition
lmax = 10 # lmax <= 15 due to PySCF
lebedev_order = 13

# Input files
density_file = 'dat/density_LiH.hdf5'
helfem_res_file = 'dat/helfem_LiH.chk'
atom_Li_file = 'dat/1e_Li_in_LiH.chk'
atom_H_file = 'dat/1e_H_in_LiH.chk'

# Read data Diatomic
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = utils.diatomic_density(density_file)
print("nuclear distance=", 2*Rh)

def inner_projection(u1, u2, dV=dV):
    return np.sum(u1 * u2 * dV)

# Transfrom HelFEM grid to cart
coords = fem.prolate_to_cart(Rh, helfem_grid)
ncoords = coords.shape[0]
print("coords shape", coords.shape)

# Reference FEM solution from HelFEM
Efem0, E2, Efem_kin, Efem_nuc, Efem_nucr = utils.diatomic_energy(helfem_res_file)
Efem = Efem0 - Efem_nucr + shift

# Adaptive basis
ibas_1, ibas_2 = 0, 0 # initialisation
max_iter = len(basis_list)
vec_Herr_adapt = []
vec_nbas1_adapt = []
vec_nbas2_adapt = []
while (True):

    basis_1 = basis_list[ibas_1]
    basis_2 = basis_list[ibas_2]
    print('\nibas_1=', ibas_1, 'ibas_2=', ibas_2)

    # Approximate GTO solution from PySCF
    mol, E_gto, C = gto.build_gto_sol(Rh, 'Li', 'H', basis_1, basis_2)
    u_gto, u_Delta_gto = gto.build_Delta(mol, coords, C)
    # Shift
    E_gto += shift
    # H(-X) = E(-X) by convention take positive
    if ( inner_projection(u_fem, u_gto) < 0 ):
        C = - C
        u_gto = - u_gto
        u_Delta_gto = - u_Delta_gto

    # True Hnorm error
    val_Delta = - 0.5 * inner_projection(u_fem, u_Delta_gto)
    dV_pot = fem.build_dV_pot(helfem_grid, Z1, Z2, wquad)
    val_pot = - np.power(Rh,2) * inner_projection(u_fem, u_gto, dV=dV_pot)
    val_ovlp = inner_projection(u_gto, u_fem)
    err_H = np.sqrt(Efem + E_gto - 2*( val_Delta + val_pot + shift * val_ovlp))

    print("H-error, adaptive=", err_H, 'Egto=', E_gto)

    # Get number of AOs per atom
    shell = mol.offset_ao_by_atom(ao_loc=None)
    nbas_1 = shell[0,1] - shell[0,0]
    nbas_2 = shell[1,1] - shell[1,0]

    # Save data for plot
    vec_Herr_adapt.append(err_H)
    vec_nbas1_adapt.append(nbas_1)
    vec_nbas2_adapt.append(nbas_2)

    """
    Compute atomic estimators
    to adapt basis for the next iteration
    """
    # Read data atomic for Li
    E_atom, orbs_rad, r_rad, w_rad = utils.atomic_energy(atom_Li_file, lmax)

    # Partition of unity evaluated on radial part
    delta = pou.delta_value(amin, amax)
    g = np.sqrt(pou.partition_vec(r_rad, amin, amax, delta))
    f = lambda xv: gto.residual(mol, xv, C, E_gto, Rh, Z1, Z2, shift)

    eigpairs = (E_atom, orbs_rad)
    rad_grid = (r_rad, w_rad)
    estim_atom_Li = norm.atom_inner(f, g, eigpairs, rad_grid, lebedev_order, lmax, shift)

    # Read data atomic for H
    E_atom, orbs_rad, r_rad, w_rad = utils.atomic_energy(atom_H_file, lmax)

    # Partition of unity evaluated on radial part
    start_t = time.time()
    delta = pou.delta_value(amin, amax)
    g = np.sqrt(pou.partition_vec(r_rad, amin, amax, delta))
    f = lambda xv: gto.residual(mol, xv, C, E_gto, Rh, Z1, Z2, shift)

    eigpairs = (E_atom, orbs_rad)
    rad_grid = (r_rad, w_rad)
    estim_atom_H = norm.atom_inner(f, g, eigpairs, rad_grid, lebedev_order, lmax, shift)
    end_t = time.time()

    print('eta_H=', estim_atom_H, 'eta_Li=', estim_atom_Li, 'time=', end_t - start_t)

    # Add more basis functions to the atom with larger estimated error
    if (estim_atom_H > estim_atom_Li):
        ibas_2 += 1
    elif (estim_atom_H < estim_atom_Li):
        ibas_1 += 1

    if (ibas_1 > max_iter - 1) or (ibas_2 > max_iter - 1):
        break

print("\nNon-adaptive basis")

# Non-adaptive basis
vec_Herr = np.empty(max_iter)
vec_nbas1 = np.empty(max_iter)
vec_nbas2 = np.empty(max_iter)
for i in range(max_iter):
    
    # Approximate GTO solution from PySCF
    basis = basis_list[i]
    mol, E_gto, C = gto.build_gto_sol(Rh, 'Li', 'H', basis, basis)
    u_gto, u_Delta_gto = gto.build_Delta(mol, coords, C)
    # Shift
    E_gto += shift
    # H(-X) = E(-X) by convention take positive
    if ( inner_projection(u_fem, u_gto) < 0 ):
        C = - C
        u_gto = - u_gto
        u_Delta_gto = - u_Delta_gto

    # True Hnorm error
    val_Delta = - 0.5 * inner_projection(u_fem, u_Delta_gto)
    dV_pot = fem.build_dV_pot(helfem_grid, Z1, Z2, wquad)
    val_pot = - np.power(Rh,2) * inner_projection(u_fem, u_gto, dV=dV_pot)
    val_ovlp = inner_projection(u_gto, u_fem)
    err_H = np.sqrt(Efem + E_gto - 2*( val_Delta + val_pot + shift * val_ovlp))

    print("H-error, non-adaptive", err_H, 'Egto=', E_gto)

    # Get number of AOs per atom
    shell = mol.offset_ao_by_atom(ao_loc=None)
    nbas_1 = shell[0,1] - shell[0,0]
    nbas_2 = shell[1,1] - shell[1,0]

    # Save results
    vec_Herr[i] = err_H
    vec_nbas1[i] = nbas_1
    vec_nbas2[i] = nbas_2

# TODO add a small timer to compute cost of atomic error indicators

Nb_list = vec_nbas1 + vec_nbas2
Nb12_list = np.array(vec_nbas1_adapt) + np.array(vec_nbas2_adapt)

plt.rcParams.update({'font.size': 15})
plt.plot(Nb_list, vec_Herr, 'g^-', label=r"$N_1=N_2$")
plt.plot(Nb12_list, vec_Herr_adapt, 'o-', color='orange', label="adaptive")
plt.ylabel("approx. error")
plt.xlabel(r"$N=N_1+N_2$ discretisation basis functions")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("img/adapt_3d.pdf")
plt.close()

