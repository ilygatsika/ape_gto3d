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
import os

"""
Main code for H-norm error estimation using practical and guaranteed estimators
Estimator is Theorem 3.7

TODO in the future this program will replace src/plot.py
"""

# User input
basis = str(sys.argv[1]) # GTO basis set
option = str(sys.argv[2]) # fine or coarse fem grid

# recover filenames
res_name, fig_name, fem_name, den_name = "res", "norm", "helfem", "density"
if (option == 'coarse'):
	res_name += "_small" 
	fig_name += "_small" 
	fem_name += "_small" 
	den_name += "_small" 
	
resfile = "out/"+res_name+".pickle"   # file to load/store results
figfile = "img/"+fig_name+".pdf"      # figure plot
femfile = "dat/"+fem_name+".chk"      # helfem input 
denfile = "dat/"+den_name+".hdf5"     # density input 
atomfile = 'dat/1e_lmax20.chk'       # spectral basis for atom

# Create out and img directory if non-existing
if (not os.path.exists("img")): os.mkdir("img") 
if (not os.path.exists("out")): os.mkdir("out")

# Load computations for this basis otherwise compute them 
try:

    with open(resfile, 'rb') as file:
	    data = pickle.load(file)

    """
    Read data
    """
    all_basis = list(data.keys())

    # check wanted basis is precomputed
    assert (basis in all_basis) 

    # get shift
    n_bas = len(all_basis)
    s = data[all_basis[0]]["shift"]
    s1 = data[all_basis[0]]["shift1"]
    s2 = data[all_basis[0]]["shift2"]
    s3 = data[all_basis[0]]["shift3"]
    shift = (s1, s2, s3, s)
    print(shift)

    estim_atom = np.array([data[all_basis[i]]["estim_atom"] for i in range(n_bas)])
    estim_Delta = np.array([data[all_basis[i]]["estim_Delta"] for i in range(n_bas)])
    estim = np.array([data[all_basis[i]]["estimator"] for i in range(n_bas)])
    err_H = np.array([data[all_basis[i]]["err_H"] for i in range(n_bas)])
    
except:

    # Parameters for partition overlap 
    amin = 0.1 # if we put larger, such as 0.5, the constant C_P exploses
    amax = 0.8 # max 0.9
    shift = 4.0 # 3.80688477
    shift_inf = 3
    sigmas = (3, 3, shift_inf, shift)
    # Parameters for spectral decomposition
    lmax = 6 # lmax <= 15 due to PySCF
    lebedev_order = 13

    # Read data Diatomic
    dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = utils.diatomic_density(denfile)

    def inner_projection(u1, u2, dV=dV):
        return np.sum(u1 * u2 * dV)

    # Transfrom HelFEM grid to cart
    coords = fem.prolate_to_cart(Rh, helfem_grid)
    ncoords = coords.shape[0]
    print("coords shape", coords.shape)

    # Reference FEM solution from HelFEM
    Efem, E2, Efem_kin, Efem_nuc, Efem_nucr = utils.diatomic_energy(femfile)
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
    cH = 1./np.sqrt(Efem)
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
    E_atom, orbs_rad, r_rad, w_rad = utils.atomic_energy(atomfile, lmax)

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
    data["denfile"] = denfile
    data["femfile"] = femfile
    data["atomfile"] = atomfile
    data["n_bas"] = mol.nbas
    key = basis
    utils.store_to_file(resfile, key, data)

"""
Plots basis size and error
"""
def main():

    # Sort with desceanding error
    idx = np.argsort(err_H)[::-1]

    """
    Plot error convergence
    TODO 
    """
    """
    plt.rcParams.update({'font.size': 13})
    plt.title(r"$\sigma_1=%.1f \sigma_2=%.1f \sigma_3=%.1f s=%.1f" %shift)
    labels = [all_basis[idx[i]] for i in range(n_bas)]
    plt.xticks(np.arange(n_bas), labels, rotation=45, fontsize=12, ha='right', rotation_mode='anchor')
    plt.plot(estim[idx], 'x-', label=r"estimator")
    plt.plot(estim_atom[idx], 'x-', label=r"atom part")
    plt.plot(estim_Delta[idx], 'x-', label=r"inf part")
    plt.plot(err_H[idx], '^-', label="approx. error")
    plt.yscale("log")
    plt.legend()
    plt.gcf().set_size_inches(7, 6)
    plt.savefig(figfile, bbox_inches='tight')
    plt.close()
    """
if __name__ == "__main__":
    main()
