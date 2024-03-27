import unittest
import numpy as np
from pyscf.solvent import ddcosmo
from pyscf.symm import sph
import src.read as read
import src.gto as gto
import src.fem as fem
import pymp

"""
Test suite
"""

# Constants
lebedev_order = 13
lmax = 6
ncores = 8

def inner_projection(u1, u2, dV):
    return np.sum(u1 * u2 * dV)

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
        eigval_H, orbs_rad, r_rad, w_rad = read.atomic_energy(atom_file, lmax)
    
        dV_rad = np.diag(np.multiply(np.square(r_rad), w_rad))
        for l in range(lmax+1):
            A = orbs_rad[l] @ dV_rad @ orbs_rad[l].T
            n = A.shape[0]
            np.testing.assert_almost_equal(A, np.eye(n))


    def test_spherical_harmonics(self):
        """
        Test orthonormalisation of spherical harmonics

        Note that low Lebedev orders give errors for high angular l
        """

        # build quadrature
        r_1sph, w_1sph = ddcosmo.make_grids_one_sphere(lebedev_order)

        ylms = sph.real_sph_vec(r_1sph, lmax, False)
        for l in range(lmax+1):

            # Array of size (m_components, npt_1sph)
            yl_vec_m = ylms[l]
            m_components = yl_vec_m.shape[0]

            # First dimension is number of spherical components
            assert( m_components == 2*l + 1 )

            # Loop on m=-l,...,l
            m_vec = np.arange(-l, l+1)
            for i_m in range(m_components):
                
                ylm = yl_vec_m[i_m]
                np.testing.assert_almost_equal(np.square(ylm) @ w_1sph, 1.000)

    def test_greens_function(self):
        """
        Test Laplacian Green's function integration using quadratures
        on Gaussian trial functions whose exact integral is known

        \int_R3 \int_R3 f(x)f(y)/|x-y| dx dy

        for f(x) = c*exp(-|r|^2) where c is normalisation factor
        """

        density_file = 'dat/density_medium_long.hdf5'
        dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = read.diatomic_density(density_file)
        coords = fem.prolate_to_cart(Rh, helfem_grid)
        ncoords = coords.shape[0]
        
        # Normalised Gaussian function
        f = lambda x: pow(1./np.pi, 3/2) * np.exp(-(x[0]**2 + x[1]**2 + x[2]**2))
        # Coulomb potential centered at zero
        V = lambda x: 1./np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

        # Evaluate 3D integral on y
        f_y_1 = pymp.shared.array(ncoords, dtype=float)
        with pymp.Parallel(ncores) as p:
            for i in p.range(ncoords):

                y = coords[i]

                # Evaluate 3D integral on z
                f_z_1 = np.array([f(y+z) for z in coords])
                f_z_2 = np.array([V(z) for z in coords])
                f_y_1[i] = inner_projection(f_z_1, f_z_2, dV)

        f_y_2 = np.array([f(y) for y in coords])
        integral = inner_projection(f_y_1, f_y_2, dV)

        # Exact value from Helgaker (bielectronic integral)
        a = 0.5 # a = pq/(p+q), p=q=1
        exact = np.sqrt(4*a/np.pi)
        # integral small:            0.7698504868876277
        # integral small long:       0.7701445190
        # integral small extra long: 0.7690315414 
        # integral medium long:      0.7698993927 
        print("Integral %.10f, exact %.10f" %(integral, exact) )

if __name__ == '__main__':
    unittest.main() 


