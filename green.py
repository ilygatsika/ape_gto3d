import src.fem as fem
import src.read as read
import src.gto as gto
import src.partition as pou
from pyscf.solvent import ddcosmo
from pyscf.symm import sph
from pyscf import dft
import numpy as np
import matplotlib.pyplot as plt
import pymp

"""
Green's function term
* paralleliser les boucles
* pour faire le résidu : ajouter le terme avec le alpha
* 1./4pi coefficient devant l'intégrale
* automatiser les tests pour lancer sur le cluster
"""

# Parameters for partition overlap 
amin = 0.1 # if we put larger, such as 0.5, the constant C_P exploses
amax = 0.8 # max 0.9
sigmas = (3, 3, 3, 9)
# Parameters for spectral decomposition
nmax = 4 # lmax <= 15 due to PySCF
shift = 4.0 # 3.80688477

# Input files
density_file = 'dat/density_small.hdf5'
helfem_res_file = 'dat/helfem_small.chk'
atom_file = 'dat/1e_lmax4_Rmax1_4.chk'

# Read data Diatomic
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = read.diatomic_density(density_file)

def inner_projection(u1, u2, dV=dV):
    return np.sum(u1 * u2 * dV)

# Transfrom HelFEM grid to cart
coords = fem.prolate_to_cart(Rh, helfem_grid)
ncoords = coords.shape[0]
print("coords shape", coords.shape)

# Reference FEM solution from HelFEM
Efem, Efem_kin, Efem_nuc, Efem_nucr = read.diatomic_energy(helfem_res_file)
Efem -= Efem_nucr
Efem += shift

# Approximate GTO solution from PySCF
basis = 'cc-pvdz'
mol, E_gto, C, E_gto_excited = gto.build_gto_sol(Rh, basis) 
ao_value = dft.numint.eval_ao(mol, coords)
u_gto = ao_value @ C
E_gto += shift
# H(-X) = E(-X) par convention on prend la positive
flag = ( inner_projection(u_fem, u_gto) < 0 )

# error should be 1.28e-01
print(basis, "eigenvalue", Efem, E_gto, "error", abs(Efem - E_gto))

"""
Laplacian estimator
takes some time
"""
"""
# vrai valeur
alpha = np.sqrt(shift)

# Integrant on y
Res = gto.residual(mol, coords, C, E_gto, Rh, Z1, Z2, flag, shift)
part_3 = pou.partition_compl(Rh, coords, amin, amax)
assert(not np.all(part_3 < 0)) # should be positive
f_y_2 = np.sqrt(part_3) * Res

def f(x): 
    expo = np.exp(-alpha * (x[0]**2 + x[1]**2 + x[2]**2))
    coulomb = 1./np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

    return expo * coulomb

# True calculation
f_y_1 = np.empty(ncoords)
for i in range(ncoords):
    
    # Grid on z
    y = coords[i]
    z_vec = coords + y

    # Integrant evaluated on z + y
    Res = gto.residual(mol, z_vec, C, E_gto, Rh, Z1, Z2, flag, shift)
    part_3 = pou.partition_compl(Rh, z_vec, amin, amax)
    f_z_1 = np.sqrt(part_3) * Res
    f_z_2 = np.array([f(z) for z in z_vec])

    # Evaluate 3D integral on z
    f_y_1[i] = inner_projection(f_z_1, f_z_2, dV)

# Evaluate 3D integral on y
factor = 1./(4*np.pi) 
estim_Delta_dual = factor * inner_projection(f_y_1, f_y_2, dV)

print(estim_Delta_dual)
"""
estim_Delta_dual = 0.1437059600650048 
# first run 0.13273287219056346

"""
Atomic estimator
"""

# Read data atomic
E_atom, orbs_rad, r_rad, w_rad = read.atomic_energy(atom_file, nmax)
num_1d = r_rad.shape[0]

# Electron-nuclear repulsion

# Load Lebedev rule on unit sphere
lmax = nmax
lebedev_order = 13
r_1sph, w_1sph = ddcosmo.make_grids_one_sphere(lebedev_order)
npt_1sph = r_1sph.shape[0]
print("Lebedev %i" %npt_1sph)

# Integration volume for radial and angular
dV_rad = np.diag(np.multiply(np.square(r_rad), w_rad))
dV_ang = np.diag(w_1sph)

# Partition of unity evaluated on radial part
p_atom = pou.partition_vec(r_rad, amin, amax)

# precompute spherical harmonics
ylms = sph.real_sph_vec(r_1sph, nmax+1, False)

estim_atom = 0 # spectral decomposition term
estim_atom_1 = 0 # remaining term
eigval_max = -1
"""
# Eigendecomposition of size nmax
for n in range(nmax+1):
 
    # Loop on l = 0,..,n-1
    for l in range(n):

        # Recover spherical harmonic
        Rln = orbs_rad[l][n]
        yl_vec_m = ylms[l]
        m_comp = yl_vec_m.shape[0]
        
        # Eigenvalue
        eigval = E_atom[l][0][n] + shift
        # store largest eigenvalue
        if (eigval > eigval_max): 
            eigval_max = eigval

        # Angular part evaluated on a grid
        int_rad = np.empty(num_1d, dtype=float)
        # fix radial part
        for i in range(num_1d):

            # Grid on r-sphere
            radius = r_rad[i]
            r_rsph = radius * r_1sph
            
            # Residual on the r-sphere
            Res = gto.residual(mol, r_rsph, C, E_gto, Rh, Z1, Z2, flag, shift)

            # Loop degeneracy on m
            val_m = np.array([Res @ dV_ang @ yl_vec_m[m].T for m in range(m_comp)])
            int_rad[i] = Rln[i] * np.sum(val_m)

        # Compute radial integral
        inter = (int_rad @ dV_rad @ np.sqrt(p_atom).T)**2
        estim_atom += 1./eigval * inter
        estim_atom_1 += inter

print(estim_atom)
"""
for n in range(nmax+1):
    int_rad = np.empty(num_1d, dtype=float)
    for i in range(num_1d):
    
        radius = r_rad[i]
        r_rsph = radius * r_1sph

        # Residual on the r-sphere
        Res = gto.residual(mol, r_rsph, C, E_gto, Rh, Z1, Z2, flag, shift)

        # Gather all associated l
        nmax = len(ylms[-1])
        val_lm = [Res @ dV_ang @ ylms[l].T for l in range(n)]
    
        # Sum over degenecary (sum over m)
        val_l = np.array([np.sum(val_lm[l]) for l in range(n)])

        # Quantity that depends on l
        for l in range(n):
        
            Rln = orbs_rad[l][n]

            # Eigenvalue
            eigval = E_atom[l][0][n] + shift
            # store largest eigenvalue
            if (eigval > eigval_max): 
                eigval_max = eigval
            
            int_rad[i] = Rln[i] * np.sum(val_l[l])

            # Compute radial integral
            integral = (int_rad @ dV_rad @ np.sqrt(p_atom).T)**2
            estim_atom += 1./eigval * integral
            estim_atom_1 += integral



print(estim_atom)

# to fix below
for n in range(nmax+1):
 
    # Loop on l = 0,..,n-1
    for l in range(n):

        # Recover spherical harmonic
        Rln = orbs_rad[l][n]
        yl_vec_m = ylms[l]
        m_comp = yl_vec_m.shape[0]
        
        # Eigenvalue
        eigval = E_atom[l][0][n] + shift
        # store largest eigenvalue
        if (eigval > eigval_max): 
            eigval_max = eigval

        # Angular part evaluated on a grid
        int_rad = np.empty(num_1d, dtype=float)
        # fix radial part
        for i in range(num_1d):
            """
            Res does not depend on l. It would be better to implement an
            external loop on num_1d that computes Res. Only ylm depends on l
            """

            # Grid on r-sphere
            radius = r_rad[i]
            r_rsph = radius * r_1sph
            
            # Residual on the r-sphere
            Res = gto.residual(mol, r_rsph, C, E_gto, Rh, Z1, Z2, flag, shift)

            # Loop degeneracy on m
            val_m = np.array([Res @ dV_ang @ ylms[l].T for m in range(n)])
            int_rad[i] = Rln[i] * np.sum(val_m)

        # Compute radial integral
        inter = (int_rad @ dV_rad @ np.sqrt(p_atom).T)**2
        estim_atom += 1./eigval * inter
        estim_atom_1 += inter

print(estim_atom)

exit()

# L2 norm of residual
# pour chaque point de vec_r il faut évaluer le résidu, car il a aussi une
# partie radiale
val_res = (Res @ dV_ang @ Res.T) * (np.sqrt(p_atom) @ dV_rad @ np.sqrt(p_atom).T)
print("DEBUG", val_res)
estim_atom_L2 = val_res

print("estimator atom", estim_atom)
print("estimator complement", estim_Delta_dual)

# Final estimator
estim_tot = 2 * (estim_atom + 1./eigval_max * (estim_atom_L2 - estim_atom_1) + estim_Delta_dual)

print("Total estimator (without constants)", estim_tot)

# Now we calculate constants
X = np.linspace(-2*Rh, 2*Rh, 500)
coords = np.zeros((X.shape[0], 3))
coords[:,2] = X
val_sup = pou.eval_supremum(a, b, Rh, Z1, Z2, sigmas)

#coords = fem.prolate_to_cart(Rh, helfem_grid)
test(coords, amin, amax, Rh, Z1, Z2, sigmas, plot=True)

#estim_Delta_dual = 0.13541234253547357

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

print("True error (H norm)", err_H)


