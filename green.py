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
* dV_pot par atomic code
* paralleliser les boucles
* pour faire le résidu : ajouter le terme avec le alpha
* 1./4pi coefficient devant l'intégrale
* automatiser les tests pour lancer sur le cluster
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
atom_file = 'dat/1e_lmax4_Rmax1_4.chk'

# Read data Diatomic
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = read.diatomic_density(density_file)

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

Efem, Efem_kin, Efem_nuc, Efem_nucr = read.diatomic_energy(helfem_res_file)
Efem -= Efem_nucr
Efem += shift

# Define PySCF Mole object
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
"""

# vrai valeur
alpha = np.sqrt(shift)

# Integrant on y
Res = gto.residual(mol, coords, C, E_gto, Rh, Z1, Z2, flag, shift)
part_3 = partition(coords)
f_y_2 = np.sqrt(part_3) * Res

def f(x): 
    expo = np.exp(-alpha * (x[0]**2 + x[1]**2 + x[2]**2))
    coulomb = 1./np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

    return expo * coulomb

#f_y_1 = np.empty(ncoords)

"""
# DEBUG Gaussians
f_y_1 = pymp.shared.array(ncoords, dtype=float)
ncore = 4
#for i in range(ncoords):
with pymp.Parallel(ncore) as p:
    for i in p.range(ncoords):
        
        y = coords[i]
    
        # Grid on z
        z_vec = coords + y

        # Integrant evaluated on z + y
        Res = gto.residual(mol, z_vec, C, E_gto, Rh, Z1, Z2, flag, shift)
        part_3 = partition(z_vec)
        f_z_1 = np.sqrt(part_3) * Res
        f_z_2 = np.array([f(z) for z in z_vec])

        # Evaluate 3D integral on z
        f_y_1[i] = inner_projection(f_z_1, f_z_2, dV)
        print(i, ncoords)

print("Final assembly..")
f_y_1 = np.array(f_y_1)
print("End assembly.")
"""

"""
# True calculation
f_y_1 = np.empty(ncoords)
for i in range(ncoords):
    
    # Grid on z
    y = coords[i]
    z_vec = coords + y

    # Integrant evaluated on z + y
    Res = gto.residual(mol, z_vec, C, E_gto, Rh, Z1, Z2, flag, shift)
    part_3 = partition(z_vec)
    f_z_1 = np.sqrt(part_3) * Res
    f_z_2 = np.array([f(z) for z in z_vec])

    # Evaluate 3D integral on z
    f_y_1[i] = inner_projection(f_z_1, f_z_2, dV)
    print(i, ncoords)

# Evaluate 3D integral on y
factor = 1./(4*np.pi) 
estim_Delta_dual = factor * inner_projection(f_y_1, f_y_2, dV)

print(estim_Delta_dual)
"""

estim_Delta_dual = 0.13273287219056346

"""
Atomic estimator
"""

# Read data atomic
E_atom, orbs, vec_r, wr = read.atomic_energy(atom_file, nmax)
num_1d = vec_r.shape[0]

# Electron-nuclear repulsion
nuc = np.array([0,0, Rh])
Vrad = lambda x: pow(x[0]**2+x[1]**2+x[2]**2,-1)
V = lambda x: Z1*Vrad(x-nuc) + Z2*Vrad(x+nuc)

# Load Lebedev rule on unit sphere
lmax = nmax
lebedev_order = 7
coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(lebedev_order)
ngrid_ang = coords_1sph.shape[0]
print("Lebedev %i" %ngrid_ang)

# unit volume for radial
dV_rad = np.diag(np.multiply(np.square(vec_r), wr))

# unit volume for angular
dV_ang = np.diag(weights_1sph)

# Partition of unity evaluated on radial part
p_atom = pou.partition_vec(vec_r, amin, amax)

estim_atom = 0 # spectral decomposition term
estim_atom_1 = 0 # remaining term
eigval_max = -1

# Eigendecomposition of size nmax
for n in range(nmax+1):
 
    # Loop on l = 0,..,n-1
    for l in range(n):

        # Radial on j and l
        """
        orbs[0][0] 1s
        orbs[0][1] 2s
        orbs[0][2] 3s
        """
        Rln = orbs[l][n]
        
        # Normalization coefficient
        int_val = Rln @ dV_rad @ Rln.T
        norml = 1./int_val
        #print(norml) is one

        # eigenvalue
        eigval = E_atom[l][0][n] + shift
        # store largest eigenvalue
        if (eigval > eigval_max): 
            eigval_max = eigval

        #print(n, l, "eigvalue", eigval)
       
        # Loop on m=-l,..,l (angular part)
        ylms = sph.real_sph_vec(coords_1sph, l)
        # scipy.special.sph_harm(-m, l, theta, varphi)
        #print("ylms", l, len(ylms))
        #print("ylms[0]", l, len(ylms[0]))
        ylm_1sph = np.vstack(ylms) # array (m,num_sph)
        num_ang = ylm_1sph.shape
        #print("angular part evaluated on grid of ", num_ang)
             
        # Angular part evaluated on a grid
        val_r = []
        # fix radial part
        for i in range(num_1d):

            # Grid on r-sphere
            r = vec_r[i]
            coords_rsph = r * coords_1sph
            
            # Residual on the r-sphere
            Res = gto.residual(mol, coords_rsph, C, E_gto, Rh, Z1, Z2, flag, shift)

            # Loop degeneracy
            val = 0
            for k in range(num_ang[0]):

                psi = ylm_1sph[k] # eigenstate at l,m
                
                # Compute angular integral associated to fixed r
                val += Rln[i] * (Res @ dV_ang @ psi.T)

            val_r.append(val)

        val_r = np.array(val_r)
        #print(n,l,val_r)
        

        # Compute radial integral
        inter = (norml * val_r @ dV_rad @ np.sqrt(p_atom).T)**2
        estim_atom += 1./eigval * inter
        estim_atom_1 += inter

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

"""
TODO calculate constants

Calculer explicitement les Delta et Grad de p_k qui est une fonction explicite
la partie radiale est nulle et après le sup sur la grille

""" 

#estim_Delta_dual = 0.13541234253547357

# True Hnorm error
# Laplacian term
lapl_ao = dft.numint.eval_ao(mol, coords, deriv=2)
u_Delta_gto = (lapl_ao[4] + lapl_ao[7] + lapl_ao[9]) @ C
val_Delta = inner_projection(u_fem, u_Delta_gto)
# Potential V (Coulomb)
val_pot = - np.power(Rh,2) * inner_projection(u_fem, u_gto, dV=dV_pot)
# Convention to take positive
if flag : 
    u_Delta_gto = - u_gto 
    u_gto = - u_gto 

# Hnorm 
err_H = np.sqrt(Efem + E_gto - 2*( - 0.5 * val_Delta - val_pot + shift))

print("True error (H norm)", err_H)


