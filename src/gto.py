from pyscf import __config__
setattr(__config__, 'B3LYP_WITH_VWN5', True)
from pyscf import gto, dft

# TODO import error
from scipy.special.constants import physical_constants
from scipy.special import assoc_laguerre

# Bohr radius
a0, _, _ = physical_constants["Bohr radius"]

"""
Call PySCF routines for GTO basis sets
"""

def build_gto_sol(Rh, basis):
    """
    Solve ground-state pb on GTOs
    """

    # Create one-electron molecule
    mol = gto.M(
            atom=f'H 0 0 {-Rh}; H 0 0 {Rh}', 
            unit='bohr', 
            charge=1, 
            spin=1,
            basis=basis, 
            verbose=0)

    # Run unrestricted Hartree-Fock
    myhf = mol.UHF()
    myhf.run()
    
    # Get ground-state energy
    E_gto_tot = myhf.kernel()
    C = myhf.mo_coeff[0][:,0]

    # Without nuclear repulsion
    E_gto = E_gto_tot - mol.energy_nuc()

    return (mol, E_gto, C)

def build_Delta(mol, coords, C):
    """
    Compute Laplacian(u_gto)
    """

    lapl_ao = dft.numint.eval_ao(mol, coords, deriv=2)
    u_Delta_gto = (lapl_ao[4] + lapl_ao[7] + lapl_ao[9]) @ C

    return u_Delta_gto

def radial_atomic(n,l,r):
    """
    Return radial part of Hydrogen wave function

    McQuarrie, page 223
    """
    # normalisation factor
    a = fact(n-l-1)
    b = 2*n*pow(fact(n+l), 3)
    c = pow(2/(n*a0), l + 3/2)
    nrml = -pow(a/b, 0.5) * c

    # exponential
    expo = pow(r,l) * np.exp(-r/(n*a0))

    # associated Laguerre polynomial L_n^k
    poly = assoc_laguerre(2/(n*a0), n+1, 2*l+1)

    return (nrml * expo * poly)

