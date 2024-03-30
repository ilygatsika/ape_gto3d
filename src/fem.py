import numpy as np

"""
Routines related to FEM solution and grid
"""

def prolate_to_cart(Rh, helfem_grid):
    """
    Transform prolate spheroidal coordinates to cartesian
    """

    mu, phi, cth = helfem_grid

    # Sth is sinus nu (nu is theta)
    sth = np.sqrt(1 - cth * cth)

    # Convert spherical (r, phi, theta) to cart (x,y,z)
    X = Rh * np.sinh(mu) * sth * np.cos(phi)
    Y = Rh * np.sinh(mu) * sth * np.sin(phi)
    Z = Rh * np.cosh(mu) * cth

    coords = np.zeros((X.shape[1], 3))
    coords[:,0] = X
    coords[:,1] = Y
    coords[:,2] = Z

    return coords

def build_dV_pot(helfem_grid, Z1, Z2, wquad):
    """
    Build integration volume element for Coulomb
    """

    mu, phi, cth = helfem_grid 
    V_Coulomb = (Z1 + Z2) * np.cosh(mu) + (Z2 - Z1) * cth
    dV_pot = V_Coulomb * np.sinh(mu) * wquad

    return dV_pot


