import src.fem as fem
import src.utils as utils
import src.gto as gto
import src.partition as pou
import src.norm as norm
import matplotlib.pyplot as plt
from numpy.linalg import norm as norm2
import numpy as np
import pickle
import sys

"""
Main code for H-norm error estimation using practical and guaranteed estimators
Estimator is Theorem 3.7
"""

try:
    with open("out/estimator_01/res.pickle", 'rb') as file:
	    data = pickle.load(file)

except: 

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
    density_file = 'dat/density_small.hdf5' # str(sys.argv[1])
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
    mol, E_gto, C = gto.build_gto_sol(Rh, 'H', 'H', basis, basis)
    u_gto, u_Delta_gto = gto.build_Delta(mol, coords, C)
    # Shift
    E_gto += shift
    # H(-X) = E(-X) by convention take positive
    if ( inner_projection(u_fem, u_gto) < 0 ):
        C = - C
        u_gto = - u_gto
        u_Delta_gto = - u_Delta_gto

    # Constant of Assumption 3
    cH = 1./Efem
    # Gap constants for the first eigenvalue
    c1 = (1 - E_gto / E2)**2 # equation 3.3, C_tilde
    c2 = (1 - E_gto / E2)**2 * E2 # equation 3.4, C_hat

    print('cH=',cH)
    print('c1=',c1)
    print('c2=',c2)


    # Constant associated to partition (equation 1.3)
    delta = pou.delta_value(amin, amax)
    val_sup = pou.eval_supremum(amin, amax, Rh, Z1, Z2, sigmas, delta)
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
    kernel = lambda x: 1./(4*np.pi) * np.exp(-alpha * norm2(x, axis=1)**2)/norm2(x, axis=1)
    # integrand
    p_res = lambda xv: np.sqrt(pou.partition_compl(Rh, xv, amin, amax, delta)) * \
            gto.residual(mol, xv, C, E_gto, Rh, Z1, Z2, shift)

    # Use green_inner_fast for small grids
    estim_Delta = norm.green_inner(p_res, kernel, coords, dV)

    print('estim_Delta=',estim_Delta)

    """
    Atomic estimator
    """

    # Read data atomic
    E_atom, orbs_rad, r_rad, w_rad = utils.atomic_energy(atom_file, lmax)

    # Partition of unity evaluated on radial part
    g = np.sqrt(pou.partition_vec(r_rad, amin, amax, delta))
    f = lambda xv: gto.residual(mol, xv, C, E_gto, Rh, Z1, Z2, shift)

    eigpairs = (E_atom, orbs_rad)
    rad_grid = (r_rad, w_rad)
    estim_atom = norm.atom_inner(f, g, eigpairs, rad_grid, lebedev_order, lmax, shift)
    print('estim_atom=', estim_atom)

    r1 = 2*estim_atom + estim_Delta

    # Now multiply by constants according to Theorem 3
    # C is cP
    # C tilde is c1
    # C hat is c2
    final_estim = pow(cP * 1./c1 * r1 + Efem * cP**2 * 1./c2**2 * r1**2, 0.5)
    print("Estimator of Theorem 3.7=", final_estim)

    # True Hnorm error
    # Laplacian term
    val_Delta = - 0.5 * inner_projection(u_fem, u_Delta_gto)
    # Potential V (Coulomb)
    dV_pot = fem.build_dV_pot(helfem_grid, Z1, Z2, wquad)
    val_pot = - np.power(Rh,2) * inner_projection(u_fem, u_gto, dV=dV_pot)
    val_ovlp = inner_projection(u_gto, u_fem)

    # Hnorm 
    err_H = np.sqrt(Efem + E_gto - 2*( val_Delta + val_pot + shift * val_ovlp))

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
    data["n_bas"] = mol.nbas
    key = basis
    utils.store_to_file(resfile, key, data)

except: 

    basis = list(data.keys())
    n_bas = len(basis)
    s = data[basis[0]]["shift"]
    s1 = data[basis[0]]["shift1"]
    s2 = data[basis[0]]["shift2"]
    s3 = data[basis[0]]["shift3"]
    shift = (s1, s2, s3, s)
    print(shift)

    estim_atom = np.array([data[basis[i]]["estim_atom"] for i in range(n_bas)])
    estim_Delta = np.array([data[basis[i]]["estim_Delta"] for i in range(n_bas)])
    estim = np.array([data[basis[i]]["estimator"] for i in range(n_bas)])
    err_H = np.array([data[basis[i]]["err_H"] for i in range(n_bas)])

    # Sort with desceanding error
    idx = np.argsort(err_H)[::-1]

    # Plot error convergence
    plt.rcParams.update({'font.size': 13})
    #plt.title(r"$\sigma_1=%.1f \sigma_2=%.1f \sigma_3=%.1f s=%.1f" %shift)
    labels = [basis[idx[i]] for i in range(n_bas)]
    plt.xticks(np.arange(n_bas), labels, rotation=45, fontsize=12, ha='right', rotation_mode='anchor')
    plt.plot(estim[idx], 'x-', label=r"estimator")
    plt.plot(estim_atom[idx], 'x-', label=r"atom part")
    plt.plot(estim_Delta[idx], 'x-', label=r"inf part")
    plt.plot(err_H[idx], '^-', label="approx. error")
    plt.yscale("log")
    plt.legend()
    plt.gcf().set_size_inches(7, 6)
    plt.savefig("img/norm.pdf", bbox_inches='tight')
    plt.close()


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


