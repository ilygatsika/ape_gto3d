import src.fem as fem
import src.utils as utils
import src.gto as gto
import src.partition as pou
from pyscf import dft
import numpy as np
from numpy.linalg import norm as norm2
import pymp
import sys
import time

"""
This code compares timings of different implementations of the residual
evaluation on multiple radial grids 
"""
# first task is to time the current implementation
def inner(u1, u2, dV):
    """
    L2 inner product with dV quadrature weights
    """
    return np.sum(u1 * u2 * dV)

def green_inner(f, kernel, coords, dV):
    """
    Return scalar equal to <f, G(x,y) f>_{L_2}
    using Green kernel discretized on coords
    """

    ncoords = coords.shape[0]

    # Evaluate 3D integral on y
    coords_transl = np.array([coords + coords[i] for i in range(ncoords)]).reshape(ncoords**2,3)
    f_vals = f(coords_transl)
    f_y_1 = np.array([inner(f_vals[i*ncoords:(i+1)*ncoords], kernel(coords), dV) for i in range(ncoords)])
    
    # Evaluate 3D integral on z
    integral = inner(f_y_1, f(coords), dV)

    return integral 

# Options
basis = str(sys.argv[1]) # GTO basis set

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
#density_file = 'dat/density.hdf5'
density_file = 'dat/density_small.hdf5'
helfem_res_file = 'dat/helfem.chk'
atom_file = 'dat/1e_lmax20_Rmax1_4.chk'

# Read data Diatomic
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = utils.diatomic_density(density_file)

def inner_projection(u1, u2, dV=dV):
    return np.sum(u1 * u2 * dV)

# Transfrom HelFEM grid to cart
coords = fem.prolate_to_cart(Rh, helfem_grid)
ncoords = coords.shape[0]

# Reference FEM solution from HelFEM
Efem, E2, Efem_kin, Efem_nuc, Efem_nucr = utils.diatomic_energy(helfem_res_file)
Efem = Efem - Efem_nucr + shift
# shift
E2 += shift

# Approximate GTO solution from PySCF
mol, E_gto, C = gto.build_gto_sol(Rh, basis)

# Up to second derivatives
#ao_value = dft.numint.eval_ao(mol, coords, deriv=2)

ao_value = dft.numint.eval_ao(mol, coords)
u_gto = ao_value @ C
# Shift
E_gto += shift
# H(-X) = E(-X) par convention on prend la positive
flag = inner_projection(u_fem, u_gto)
if ( flag < 0 ):
    u_gto = - u_gto

# error should be 1.28e-01
print(basis, "eigenvalue", Efem, E_gto, "error", abs(Efem - E_gto))


"""
Laplacian estimator
takes some time
"""

# Green's function of the screened Laplacian operator
alpha = np.sqrt(shift_inf)
kernel = lambda x: 1./(4*np.pi) * np.exp(-alpha * norm2(x, axis=1)**2)/norm2(x, axis=1)
# integrand
#p_res = lambda xv: np.sqrt(pou.partition_compl(Rh, xv, amin, amax)) * \
#        gto.residual(mol, xv, C, u_fem, E_gto, flag, Rh, Z1, Z2, shift)
delta = pou.delta_value(amin, amax)
#p_res = lambda xv: np.sqrt(pou.partition_compl(Rh, xv, amin, amax, delta))
#p_res = lambda xv: np.linalg.norm(xv)
p_res = lambda xv: gto.residual(mol, xv, C, u_fem, E_gto, flag, Rh, Z1, Z2, shift)

start = time.time()
estim_Delta = green_inner(p_res, kernel, coords, dV)
end = time.time()

print('estim_Delta=',estim_Delta)
print('Total time %.2f seconds' %(end-start) )




