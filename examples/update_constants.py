import src.utils as utils
import src.gto as gto
import numpy as np
import src.partition as pou
import pickle
import sys

"""
This code overwrites all constants in res.pickle results
as well as impacted estimators
"""

# User input
basis = str(sys.argv[1]) # GTO basis set

femfile = "dat/helfem.chk"
resfile = "out/res.pickle"   
denfile = "dat/density.hdf5"

with open(resfile, 'rb') as file:
    data = pickle.load(file)
    estim_atom = data[basis]["estim_atom"]
    estim_Delta = data[basis]["estim_Delta"]
    shift = data[basis]["shift"]
    shift1 = data[basis]["shift1"]
    shift2 = data[basis]["shift2"]
    shift3 = data[basis]["shift3"]
    amin = data[basis]["amin"]
    amax = data[basis]["amax"]


# Read data Diatomic
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = utils.diatomic_density(denfile)

# Reference FEM solution from HelFEM
Efem, E2, Efem_kin, Efem_nuc, Efem_nucr = utils.diatomic_energy(femfile)
Efem = Efem - Efem_nucr + shift
E2 += shift

# Approximate solution on GTOs
mol, E_gto, E_gto_2, C = gto.build_gto_sol(Rh, 'H', 'H', basis, basis)
E_gto += shift
E_gto_2 += shift

print("exact E2=", E2, "gto E2=", E_gto_2)

# Constant of Assumption 3
cH_exact = 1./np.sqrt(Efem) # exact
low_bound = - 2 + shift # this works for H2+
cH = 1./np.sqrt(low_bound) # practical

print("exact cH=", cH_exact, "practical cH=", cH)

# Gap constants gamma_1
c1_exact = (1 - E_gto / E2)**2
c2_exact = (1 - E_gto / E2)**2 * E2
# now the practical ones
c1 = (1 - E_gto / E_gto_2)**2
c2 = (1 - E_gto / E_gto_2)**2 * E_gto_2

print("exact c1=", c1_exact, "practical c1=", c1)
print("exact c2=", c2_exact, "practical c2=", c2)

# Constant associated to partition (equation 1.3)
delta = pou.delta_value(amin, amax)
sigmas = (shift1, shift2, shift3, shift)
val_sup = pou.eval_supremum(amin, amax, Rh, Z1, Z2, sigmas, delta)
cP = 1 + cH**2 * val_sup

r1 = 2*estim_atom + estim_Delta
final_estim = pow(cP * 1./c1 * r1 + Efem * cP**2 * 1./c2**2 * r1**2, 0.5)

print("cP=", cP)
print("Estimator of Theorem 3.7=", final_estim)

# do not forget the eigenvalue bound
estim_eigval = cP * 1./c1 * r1 

# Store results to file
data = {}
data["cP"] = cP
data["cH"] = cH
data["c1"] = c1
data["c2"] = c2
data["estimator"] = final_estim
data["estim_eigval"] = estim_eigval
key = basis
utils.store_to_file(resfile, key, data)


