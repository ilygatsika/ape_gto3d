import unittest
import numpy as np
from pyscf.solvent import ddcosmo
from pyscf.symm import sph
import src.read as read
import src.gto as gto

"""
Test suite
"""

def tfunc(r):
    """
    Trial function to test
    Source: PyLebedev package
    
    Adapted from: https://cbeentjes.github.io/files/Ramblings/QuadratureSphere.pdf
    
    This function has the exact result upon integration over a unit sphere
    of 216/35 * pi
    """
    return 1 + r[0] + r[1]**2 + r[0]**2*r[1] + r[0]**4 + r[1]**5 + r[0]**2*r[1]**2*r[2]**2

class Test(unittest.TestCase):

    def test_lebedev(self):
        """
        Test Lebedev quadrature for probe function
        """

        # build quadrature
        lebedev_order = 7
        r_1sph, w_1sph = ddcosmo.make_grids_one_sphere(lebedev_order)
        npt_1sph = r_1sph.shape[0]

        # coords lie on the unit sphere
        np.testing.assert_almost_equal(np.linalg.norm(r_1sph, axis=1), np.ones(npt_1sph))

        # exact answer to function "testfunc"
        exact = 216.0 * np.pi / 35.0
        
        # Lebedev quadrature
        fval_1sph = np.apply_along_axis(tfunc, -1, r_1sph)
        integral = fval_1sph @ w_1sph

        print('Lebedev error: %.10e' % (integral-exact))
        np.testing.assert_almost_equal(exact, integral)

    def test_hydrogen_eigenstates_radial(self):
        """
        Test radial part of Hydrogen eigenstates obtained by HelFEM
        """

        atom_file = 'dat/1e_lmax20.chk'
        lmax = 15
        eigval_H, orbs_rad, r_rad, w_rad = read.atomic_energy(atom_file, lmax)
    
        dV_rad = np.diag(np.multiply(np.square(r_rad), w_rad))
        for l in range(lmax+1):
            A = orbs_rad[l] @ dV_rad  @ orbs_rad[l].T
            n = A.shape[0]
            np.testing.assert_almost_equal(A, np.eye(n))


    def test_spherical_harmonics(self):
        """
        Test orthonormalisation of spherical harmonics
        """

        lebedev_order = 7
        r_1sph, w_1sph = ddcosmo.make_grids_one_sphere(lebedev_order)

        lmax = 14
        for l in range(lmax+1):
            ylms = sph.real_sph_vec(r_1sph, l)
            for m in range(-l, l+1):
                orbs_sph = gto.get_ylm(l, m, r_1sph)

                np.testing.assert_almost_equal(np.square(orbs_sph) @ w_1sph, 1.000)
                #np.testing.assert_almost_equal(np.square(ylms[l]) @ w_1sph, 1.000)

 
if __name__ == '__main__':
    unittest.main() 

