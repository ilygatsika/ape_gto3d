from math import exp, log
import numpy as np
import h5py
import matplotlib.pyplot as plt
import src.fem as fem

#from src.partition import partition_vec
import src.read as read

# Deactivate PySCF error message
from pyscf import __config__
setattr(__config__, 'B3LYP_WITH_VWN5', True)
from pyscf import gto, dft
from pyscf.symm import sph
from pyscf.solvent import ddcosmo

"""
Main code for H-norm error estimation using practical and guaranteed estimators
"""

# Input files
density_file = 'dat/density.hdf5'
helfem_res_file = 'dat/helfem.chk'

# Read data
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2, T_fem = read.diatomic_density(density_file)
Efem, Efem_kin, Efem_nuc, Efem_nucr = read.diatomic_energy(helfem_res_file)

# Efem = Efem_kin + Efem_nuc + Efem_nucr

####################
# Helper functions #
####################
def inner_projection(u1, u2, dV=dV):

    return np.sum(u1 * u2 * dV)

# real_sph_vec(r, lmax, reorder_p=False)
# Real spherical harmonics up to the angular momentum lmax
# spherical harmonics are calculated in PySCF using cartesian to spherical trans
lmax = 2
lebedev_order = 7
#gen_atomic_grids(prune=None)
coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(lebedev_order)
# coords_1sph contains points (x,y,z) on the unit sphere
print(np.linalg.norm(coords_1sph, axis=1))
print(np.linalg.norm(coords_1sph, axis=1).shape)

print(coords_1sph.shape)
print(weights_1sph.shape)

def evaluate_sph(f,r,phi,theta):
    
    g1 = r * np.sin(phi) * np.cos(theta)
    g2 = r * np.sin(phi) * np.sin(theta)
    g3 = r * np.cos(phi)

    return (lambda phi, theta: f(gr,gphi,gtheta))

coords = fem.prolate_to_cart(Rh, helfem_grid)
ylms = sph.real_sph_vec(coords_1sph, lmax, True)
ylm_1sph = np.vstack(ylms)
print(ylm_1sph.shape)

# TODO
# Pour obtenir la grille il faut lancer le calcul 1e et puis dans helfem.chk
# il faut multiplier les coordonnées par le radius
# waiting for vec_r from Susi
'''
p = partition_vec(vec_r, a, b)
for i in range(n):
    r = vec_r[i]
    grid = r * coords_1sph
    # angular quadrature for each r
'''

mu, phi, cth = helfem_grid 
# Test equation 22
dV_test = np.power(Rh, 3) * np.sinh(mu) * (np.cosh(mu)**2 - cth**2) * wquad
print(all(np.isclose(dV.flatten(), dV_test.flatten())))

# Sth is sinus nu (nu is theta)
sth = np.sqrt(1 - cth * cth)

# Coulomb potential without singularity in (mu,nu) coordinates
V_Coulomb = (Z1 + Z2) * np.cosh(mu) + (Z2 - Z1) * cth
dV_pot = V_Coulomb * np.sinh(mu) * wquad

# Debug : this should reproduce results of Nuclear repulsion from diatomic
val_pot = - np.power(Rh,2) * inner_projection(u_fem, u_fem, dV=dV_pot)
print("Debug fem fem pot", np.isclose(val_pot,  Efem_nuc))

# Reference energy with FEM
E_fem = Efem_kin + Efem_nuc
print(E_fem) 

# Transform prolate spheroidal coordinates to cartesian
X = Rh * np.sinh(mu) * sth * np.cos(phi)
Y = Rh * np.sinh(mu) * sth * np.sin(phi)
Z = Rh * np.cosh(mu) * cth


coords = np.zeros((X.shape[1], 3))
coords[:,0] = X
coords[:,1] = Y
coords[:,2] = Z

print(X.shape)

# Devrait être 1
print("fem fem ", inner_projection(u_fem, u_fem) )

# pc is polarization consistent
bases = ["cc-pvdz", "unc-cc-pvdz", "unc-cc-pvtz", "pc-1", "unc-pc-1",
         "pc-2", "unc-pc-2", "cc-pvtz", "aug-cc-pvdz", "aug-cc-pvtz",
         "aug-cc-pvqz", "cc-pvqz", "cc-pv5z", "aug-cc-pv5z", "pc-3", "pc-4",
         "aug-pc-3", "aug-pc-4", "unc-pc-4"]

err_eigval = []
err_eigvec_L2 = []
err_eigvec_H = []
bas_size = []

for basis in bases: 

    print("\n", basis)
    mol = gto.M(atom=f'H 0 0 {-Rh}; H 0 0 {Rh}', unit='bohr', charge=1, spin=1,
            basis=basis, verbose=0)
    myhf = mol.UHF()
    myhf.run()
    E_gto_tot = myhf.kernel()
    bas_size.append(mol.nbas)

    C = myhf.mo_coeff[0][:,0]
    
    # Debug
    S = myhf.get_ovlp()
    Smo = C.T @ S @ C
    print("Debug electron number", np.isclose(Smo, 1.0))

    # Returns array of shape (N,nao)
    ao_value = dft.numint.eval_ao(mol, coords)
    #print(ao_value.shape)
    u_gto = ao_value @ C

    # H(-X) = E(-X) par convention on prend la positive
    if inner_projection(u_fem, u_gto) < 0 :
        u_gto = - u_gto 

    # Returns array of shape (M,N,nao)
    # where N number of grids, nao number of AOs, M=10 for deriv=2
    # the first (N,nao) elements are the AO values, followed by
    # 2nd derivatives (6,N,nao) for xx, xy, xz, yy, yz, zz
    lapl_ao = dft.numint.eval_ao(mol, coords, deriv=2)
    u_Delta_gto = (lapl_ao[4] + lapl_ao[7] + lapl_ao[9]) @ C
    Ekin = - 0.5 * inner_projection(u_gto, u_Delta_gto)

    # Compute residual
    # coordinates should be given by the radian part
    
    #res_gto = E_gto * u_gto - (u_Delta_gto - V*u_gto)
    

    # reference kinetic energy from PySCF
    T = mol.intor_symmetric('int1e_kin')
    Ekin_ref = C.T @ T @ C

    print("Debug kinetic", np.isclose(np.abs(Ekin_ref), np.abs(Ekin)))

    # Debug electron-nuclear attraction for gto vs gto
    Enuc = - np.power(Rh,2) * inner_projection(u_gto, u_gto, dV=dV_pot)

    # reference potential from PySCF
    V = mol.intor('int1e_nuc')
    Enuc_ref = C.T @ V @ C
    #print("Debug potential", np.isclose(Enuc_ref, Enuc))

    #print("Tot gto", -(Ekin - Enuc), E_gto_tot - mol.energy_nuc())
    #print("Tot fem", E_fem)

    # Si la grille FEM est ok les deux devraient etre 1
    #print("gto gto ", inner_projection(u_gto, u_gto) )
    #print("fem fem ", inner_projection(u_fem, u_fem) )

    # Si proche de 1 l'approximation est bonne
    # norme de la différence en carrée avec L2
    # integral \langle u_fem, u_gto\rangle
    err_l2 = inner_projection(u_gto,u_gto) + inner_projection(u_fem, u_fem) - \
            2*inner_projection(u_fem, u_gto)
    print("Erreur u_fem - u_gto en norme L2  %.2e" % err_l2 )

    # Laplacian term
    # integral \langle u_fem, Delta u_gto\rangle
    val_Delta = -0.5 * inner_projection(u_fem, u_Delta_gto)
    #print("Laplacian term ", val_Delta)


    # Potential V (Coulomb)
    val_pot = - np.power(Rh,2) * inner_projection(u_fem, u_gto, dV=dV_pot)
    #print("Potential term ", val_pot)

    # Get GTO total energy
    E_gto = - (Ekin - Enuc)

    err_H = E_fem + E_gto - 2*( - val_Delta + val_pot)
    print("Erreur u_fem - u_gto en norme H %.2e" % err_H )
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
plt.xticks(np.arange(ntest), labels, rotation=45, fontsize=8, ha='right', rotation_mode='anchor')
plt.plot(err_eigval, 'o-', label=r"$|\lambda_1 - \lambda_{1N}|$")
plt.plot(err_eigvec_L2, 'x-', label=r"$u_1 - u_{1N}$ in L2")
plt.plot(err_eigvec_H, '^-', label=r"$u_1 - u_{1N}$ in H")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("img/norms.pdf")
plt.close()




