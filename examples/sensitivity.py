import src.fem as fem
import src.utils as utils
import src.gto as gto
import src.partition as pou
import src.norm as norm
from numpy.linalg import norm as norm2
import numpy as np
import sys

"""
Parameter sensibility analysis.
Produces data for preconstant variation for varying choice of shift and
overlap width. For H2^{1+} molecule.

This program outputs latex tables.

Article notation <-> code notation
CH       As. 3   <-> cH
C tilde eq 3.3   <-> c1
C hat   eq 3.4   <-> c2
C       eq 3.1   <-> cP
"""

# Options
basis = str(sys.argv[1]) # GTO basis set

# Parameters for partition overlap 
amin = 0.1 # if we put larger, such as 0.5, the constant C_P exploses
amax = 0.8 # max 0.9                # this should be a vector
shift = 4.0 # 3.80688477            # this should be a vector
shift_inf = 3                       # this should be a vector
sigmas = (3, 3, shift_inf, shift)   # this should be a vector

# Input files
density_file = 'dat/density.hdf5'
helfem_res_file = 'dat/helfem.chk'

# Read data Diatomic
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = utils.diatomic_density(density_file)

# Reference FEM solution from HelFEM
Efem0, E20, Efem_kin, Efem_nuc, Efem_nucr = utils.diatomic_energy(helfem_res_file)
Efem = Efem0 - Efem_nucr + shift
# shift
E2 = E20 + shift

# Approximate GTO solution from PySCF
mol, E_gto0, C = gto.build_gto_sol(Rh, 'H', 'H', basis, basis)
# Shift
E_gto = E_gto0 + shift

# Constant of Assumption 3
cH = 1./np.sqrt(Efem)
# Gap constants for the first eigenvalue
c1 = (1 - E_gto / E2)**2 # equation 3.3, C_tilde
c2 = (1 - E_gto / E2)**2 * E2 # equation 3.4, C_hat

print(Rh, amin, amax)

# Constant associated to partition (equation 3.1)
for shift_atom in [1.0, 2.0, 3.0, 4.0, 5.0]:

    sigmas = (shift_atom, shift_atom, shift_inf, shift)
   
    for amax in [0.2, 0.4, 0.6, 0.8, 0.89]:
    
        # define amin, amax
        delta = pou.delta_value(amin, amax)
        ell = amax - amin # overlap width
        val_sup = pou.eval_supremum(amin, amax, Rh, Z1, Z2, sigmas, delta)
        cP = 1 + cH**2 * val_sup

        print('sigma_a=', shift_atom, 'ell=', ell, 'cP=', cP)

# Gap table
print(r'$\sigma$ & $\lambda_1/\lambda_2$ & $c_H$ \\ \hline')
for shift in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:

    sigmas = (3, 3, shift_inf, shift)
    Efem = Efem0 - Efem_nucr + shift
    E2 = E20 + shift
    cH = 1./np.sqrt(Efem)
    print(shift, ' & ', Efem / E2, ' & ', cH, r'\\')



