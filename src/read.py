import h5py
import numpy as np

"""
Routines for reading FEM solution data from HelFEM output files
"""

def diatomic_density(filename):
    """
    Read data from density file
    """

    f1 = h5py.File(filename)
    dV = np.array(f1["dV"])
    Rh = np.array(f1["Rh"])
    phi = np.array(f1["phi"])
    mu = np.array(f1["mu"])
    cth = np.array(f1["cth"])
    wquad = np.array(f1["wquad"]) # wquad = dmu *  dOmega
    u_fem = np.array(f1["orba.re"])
    Z1 = np.array(f1["Z1"]) # Z1 noyau à gauche
    Z2 = np.array(f1["Z2"]) # Z2 noyau à droite

    fem_grid = (mu, phi, cth)
    
    return (dV, Rh, fem_grid, wquad, u_fem, Z1, Z2)

def diatomic_energy(filename):
    """
    Read data from energy file
    """

    f2 = h5py.File(filename)
    Efem = np.array(f2["Etot"]) 
    Efem_kin = np.array(f2["Ekin"]) 
    Efem_nuc = np.array(f2["Enuc"]) 
    Efem_nucr = np.array(f2["Enucr"])
    E2 = np.array(f2['Ea'])[0][1]

    return (Efem, E2, Efem_kin, Efem_nuc, Efem_nucr)

def atomic_energy(filename, lmax):
    """
    Read excited states of radial problem
    """

    f3 = h5py.File(filename)

    # eigenvalues and eigevectors, angular part
    E = []
    orbs = []
    for l in range(lmax + 1):
        E.append(np.array(f3[f'E_{l}']))
        orbs.append(np.array(f3[f'orbs_{l}']))

    # Radial part evaluated on quadrature points
    r = np.array(f3["r"]).flatten()

    # quadrature weights
    wr = np.array(f3["wr"]).flatten()

    return (E, orbs, r, wr)

