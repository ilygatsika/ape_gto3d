from pylebedev import PyLebedev
import numpy as np

"""
PySCF
mf.xc='B3LYP'
mf.grids.atom_grid=(50,194)
mf.grids.becke_scheme=dft.gen_grid.original_becke
mf.grids.prune=None
mf.grids.radi_method=dft.radi.gauss_chebyshev
mf.grids.radii_adjust=None
mf.grids.verbose=4
mf.small_rho_cutoff=1e-10

mf.kernel()

Quadrature Cartesian 3D
f = lambda x, y, z: np.exp(-(x ** 2 + y ** 2 + z ** 2))
integrate.tplquad(f, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)
"""

def main():
    """
    Test Lebedev quadrature for probe function
    """
    # build library
    leblib = PyLebedev()
    
    # exact answer to function "testfunc"
    exact = 216.0 * np.pi / 35.0
    r,w = leblib.get_points_and_weights(9)
    integral = 4.0 * np.pi * np.sum(w * tfunc(r[:,0], r[:,1], r[:,2]))
    
    print('Integral: %f vs Exact: %f' % (integral, exact))

def tfunc(x,y,z):
    """
    Trial function to test
    
    Adapted from: https://cbeentjes.github.io/files/Ramblings/QuadratureSphere.pdf
    
    This function has the exact result upon integration over a unit sphere
    of 216/35 * pi
    """
    return 1 + x + y**2 + x**2*y + x**4 + y**5 + x**2*y**2*z**2

if __name__ == '__main__':
    main()
