import src.fem as fem
import src.read as read
import src.gto as gto
import src.partition as pou
from pyscf.solvent import ddcosmo
import numpy as np

# Input files
density_file = 'dat/density.hdf5'
helfem_res_file = 'dat/helfem.chk'
atom_file = 'dat/1e_lmax6.chk'

# Read data Diatomic
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = read.diatomic_density(density_file)
Efem, Efem_kin, Efem_nuc, Efem_nucr = read.diatomic_energy(helfem_res_file)

print(Rh) 

# Read data Atomic
E, orbs, vec_r, wr = read.atomic_energy(atom_file)
num_1d = vec_r.shape[0]
print("Using 1D grid of %i points" % num_1d)

print(np.max(vec_r))
exit()

# Transfrom HelFEM grid to cart
coords = fem.prolate_to_cart(Rh, helfem_grid)

# Define PySCF Mole object
basis = 'cc-pvdz'
mol, E_gto, C = gto.build_gto_sol(Rh, basis) 

# Task 1: compute the inverse dual norm of the residual
# must compute discretized Laplacian on fem
# then solve the problem with rhs


# Electron-nuclear repulsion
Vrad = lambda x: pow(x[0]**2+x[1]**2+x[2]**2,-1)
V = lambda x: Z1*Vrad(x-Rh) + Z2*Vrad(x+Rh)

#f = pou.partition_vec(r, a, b)


exit()

# Task 2: compute the inverse dual atomic norm of the residual

# Load Lebedev rule on unit sphere
lmax = 2
lebedev_order = 7
coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(lebedev_order)

nmax = 3

# Loop over radial parts
for i in range(num_1d):

    # Radial part
    r = vec_r[i]
    # Define grid on r-sphere
    grid = r * coords_1sph

    # Kinetic term evaluated on grid
    u_Delta_gto = gto.build_Delta(mol, grid, C)

    # Eigenbasis of size nmax
    eigfun = 0
    # Loop n = 1,2,3,..,nmax
    for n in range(1,nmax+1):

        # Loop l=0,1,2,..,n-1
        for l in range(n):

            # radial part
            rad_nl = gto.radial_atomic(n,l,r)

            # Loop m=-l,..,l (degeneracy)
            Ylm= 0
            for m in range(-l, l+1):
                # spherical harmonic Y_l^m
                Ylm += orbs[m][l]

        # Eigenfunction (radial*ang)
        eigfun += rad_nl * Ylm


