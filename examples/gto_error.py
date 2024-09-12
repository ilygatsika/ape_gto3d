# Plot
from math import exp, log
import numpy as np
import h5py
import matplotlib.pyplot as plt
import src.fem as fem
import src.norm as norm

#from src.partition import partition_vec
import src.utils as utils

# Deactivate PySCF error message
from pyscf import __config__
setattr(__config__, 'B3LYP_WITH_VWN5', True)
from pyscf import gto, dft

"""
Routines for plotting results
"""

# Input files
result_file = "dat/res.npy"
density_file = 'dat/density_small.hdf5'
helfem_res_file = 'dat/helfem_small.chk'

# Read data
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = utils.diatomic_density(density_file)
Efem, E2, Efem_kin, Efem_nuc, Efem_nucr = utils.diatomic_energy(helfem_res_file)

coords = fem.prolate_to_cart(Rh, helfem_grid)
dV_pot = fem.build_dV_pot(helfem_grid, Z1, Z2, wquad)

print("nuclear distance=", Rh)

# Reference energy with FEM
# Efem = Efem_kin + Efem_nuc + Efem_nucr
E_fem = Efem_kin + Efem_nuc

# pc is polarization consistent
"""bases = ["cc-pvdz", "unc-cc-pvdz", "unc-cc-pvtz", "pc-1", "unc-pc-1",
         "pc-2", "unc-pc-2", "cc-pvtz", "aug-cc-pvdz", "aug-cc-pvtz",
         "aug-cc-pvqz", "cc-pvqz", "cc-pv5z", "aug-cc-pv5z", "pc-3", "pc-4",
         "aug-pc-3", "aug-pc-4", "unc-pc-4"]
"""
bases = ["cc-pvdz","aug-cc-pvtz"]

err_eigval = []
err_eigvec_L2 = []
err_eigvec_H = []
bas_size = []

for basis in bases: 

    print("\n", basis)

    # GTO solution
    mol, E_gto_tot, C = gto.build_gto_sol(Rh, basis)
    bas_size.append(mol.nbas)

    # Returns array of shape (N,nao)
    ao_value = dft.numint.eval_ao(mol, coords)
    u_gto = ao_value @ C

    # H(-X) = E(-X) par convention on prend la positive
    if norm.inner(u_fem, u_gto, dV) < 0 :
        u_gto = - u_gto 

    #print("Tot gto", -(Ekin - Enuc), E_gto_tot - mol.energy_nuc())
    #print("Tot fem", E_fem)

    # integral \langle u_fem, u_gto\rangle
    err_l2 = 2*(1 - norm.inner(u_fem, u_gto, dV))
    print("Erreur u_fem - u_gto en norme L2  %.2e" % err_l2 )

    # Laplacian term
    # integral \langle u_fem, Delta u_gto\rangle
    u_Delta_gto = gto.build_Delta(mol, coords, C)
    val_Delta = -0.5 * norm.inner(u_fem, u_Delta_gto, dV)
    val_pot = - np.power(Rh,2) * norm.inner(u_fem, u_gto, dV_pot)
    E_gto = - (Ekin - Enuc)

    err_H = E_fem + E_gto - 2*( - val_Delta + val_pot)
    print("Erreur u_fem - u_gto en norme H %.2e" % err_H )
    print(E_fem, E_gto)
    print("Erreur on eigenvalue %.2e" % abs(E_fem - E_gto))

    # Store results to lists
    err_eigval.append(abs(E_fem - E_gto))
    err_eigvec_L2.append(np.sqrt(err_l2))
    err_eigvec_H.append(np.sqrt(err_H))

# Sort regarding eigval error
idx = np.argsort(err_eigval)[::-1]
err_eigval = [err_eigval[i] for i in idx]
err_eigvec_L2 = [err_eigvec_L2[i] for i in idx]
err_eigvec_H = [err_eigvec_H[i] for i in idx]
bases = [bases[i] for i in idx]
bas_size = [bas_size[i] for i in idx]

# Plot error convergence
ntest = len(bases)
labels = [str(bas_size[i]) + ' (' + bases[i] + ')' for i in range(ntest)]

plt.rcParams.update({'font.size': 15})
plt.xticks(np.arange(ntest), labels, rotation=45, fontsize=8, ha='right', rotation_mode='anchor')
plt.plot(err_eigval, 'o-', label=r"$|\lambda_1 - \lambda_{1N}|$")
plt.plot(err_eigvec_L2, 'x-', label=r"$u_1 - u_{1N}$ in L_2")
plt.plot(err_eigvec_H, '^-', label=r"$u_1 - u_{1N}$ in H")
plt.plot([5.36236199475814,5.338068347689154], '-r*', label="estimator")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("img/norms.pdf")
plt.close()
