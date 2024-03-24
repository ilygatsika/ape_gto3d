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

    X = Rh * np.sinh(mu) * sth * np.cos(phi)
    Y = Rh * np.sinh(mu) * sth * np.sin(phi)
    Z = Rh * np.cosh(mu) * cth

    coords = np.zeros((X.shape[1], 3))
    coords[:,0] = X
    coords[:,1] = Y
    coords[:,2] = Z

    return coords

