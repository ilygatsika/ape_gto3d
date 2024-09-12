from pyscf import __config__
setattr(__config__, 'B3LYP_WITH_VWN5', True)
from pyscf import gto, dft
from scipy import constants
from scipy.special import assoc_laguerre
from math import factorial as fact
import numpy as np

# Bohr radius
a0,_,_ = constants.physical_constants['Bohr radius']

"""
Call PySCF routines for GTO basis sets
"""

z_dic = {'H': 1, 'He': 2, 'Li': 3}

def build_gto_sol(Rh, elem1, elem2, basis1, basis2):
    """
    Solve ground-state pb on GTOs
    allows diatomics of different atoms
    """

    # Create one-electron molecule
    mol = gto.M(
            atom=f'{elem1} 0 0 {-Rh}; {elem2} 0 0 {Rh}', 
            unit='bohr', 
            charge=z_dic[elem1] + z_dic[elem2] - 1, 
            spin=1,
            basis={elem1: basis1, elem2: basis2}, 
            verbose=0
            )

    # Run unrestricted Hartree-Fock
    myhf = mol.UHF()
    myhf.run()
    
    # Get ground-state energy
    E_gto_tot = myhf.kernel()
    C = myhf.mo_coeff[0][:,0]

    # Without nuclear repulsion
    E_gto = E_gto_tot - mol.energy_nuc()
    
    # Excited states (10 for H2)
    #E_gto_excited = myhf.mo_energy[0]
    
    #return (mol, E_gto, C, E_gto_excited)
    return (mol, E_gto, C)

def build_Delta(mol, coords, C):
    """
    Compute Laplacian(u_gto)
    """

    # Returns array of shape (M,N,nao)
    # where N number of grids, nao number of AOs, M=10 for deriv=2
    # the first (N,nao) elements are the AO values, followed by
    # 2nd derivatives (6,N,nao) for xx, xy, xz, yy, yz, zz
    lapl_ao = dft.numint.eval_ao(mol, coords, deriv=2)

    u_gto = lapl_ao[0] @ C
    u_Delta_gto = (lapl_ao[4] + lapl_ao[7] + lapl_ao[9]) @ C

    return (u_gto, u_Delta_gto)

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

def Coulomb(Rh, Z1, Z2):
    """
    Electron-nuclear repulsion
    Coulomb potential as a function
    """

    nuc = np.array([0,0, Rh])
    Vrad = lambda x: 1./np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    V = lambda x: Z1*Vrad(x-nuc) + Z2*Vrad(x+nuc)

    return V

def residual(mol, coord, C, E_gto, Rh, Z1, Z2, shift):
    """
    Compute residual of Gaussian discretisation
    flag takes care of sign convention
    """

    # Kinetic term
    u_gto, u_Delta_gto = build_Delta(mol, coord, C)

    # Coulomb term
    V = Coulomb(Rh, Z1, Z2)
    V_eval = [V(point) for point in coord]
    u_V = np.multiply(V_eval, u_gto)

    # Hu term *do not forget the shift*
    Hu_gto = -0.5 * u_Delta_gto - u_V + shift * u_gto
    
    return (E_gto * u_gto - Hu_gto) 


