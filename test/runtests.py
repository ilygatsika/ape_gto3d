import unittest
import numpy as np
from pyscf.solvent import ddcosmo
from pyscf.symm import sph
from pyscf import dft
import src.utils as utils
import src.gto as gto
import src.fem as fem
import src.partition as pou
import src.norm as norm
import matplotlib.pyplot as plt
import os

"""
Test suite
"""

# Constants
lebedev_order = 13
lmax = 6 # 6 for H2+ and 10 for LiH^{3+}
         # note there is a small error in spherical harmonics due to this
         # on the fourth decimal the failing assertion is 
         # np.testing.assert_almost_equal(np.square(ylm) @ w_1sph, 1.000)
         # Arrays are not almost equal to 7 decimals
         # ACTUAL: 1.0003004807692315
         # DESIRED: 1.0
         #
basis = "aug-cc-pvtz"

def tfunc(r):
    """
    Trial function to test

    Source: PyLebedev package 
    Adapted directly from: pylebedev/tests/test_lebedev_quadrature.py
    
    This function has the exact result upon integration over a unit sphere
    of 216/35 * pi
    """
    return 1 + r[0] + r[1]**2 + r[0]**2*r[1] + r[0]**4 + r[1]**5 + r[0]**2*r[1]**2*r[2]**2

class Test(unittest.TestCase):

    def test_FEM_energy_terms(self):
        """
        Obtain FEM energies from FEM quadratures
        """

        # Input files
        density_file = 'dat/density.hdf5'
        helfem_res_file = 'dat/helfem.chk'

        # Read data
        dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = utils.diatomic_density(density_file)
        Efem, E2, Efem_kin, Efem_nuc, Efem_nucr = utils.diatomic_energy(helfem_res_file)

        # FEM solution is normalized
        np.testing.assert_almost_equal(norm.inner(u_fem, u_fem, dV), 1.00)

        # Test equation 22 of Susi's paper
        mu, phi, cth = helfem_grid 
        dV_test = np.power(Rh, 3) * np.sinh(mu) * (np.cosh(mu)**2 - cth**2) * wquad
        np.testing.assert_almost_equal(dV.flatten(), dV_test.flatten())

        # Nuclear repulsion energy term
        dV_pot = fem.build_dV_pot(helfem_grid, Z1, Z2, wquad)
        val_pot = - Rh**2 * norm.inner(u_fem, u_fem, dV_pot)
        np.testing.assert_almost_equal(val_pot, Efem_nuc)

    def test_GTO_energy_terms(self):
        """
        Obtain GTO energies from FEM quadratures
        """
       
        # Load FEM quadrature 
        density_file = 'dat/density.hdf5'
        dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = utils.diatomic_density(density_file)
        coords = fem.prolate_to_cart(Rh, helfem_grid)
        
        # Build PySCF molecule
        mol, E_gto, C = gto.build_gto_sol(Rh, 'H', 'H', basis, basis)
        
        # Debug electron number
        S = mol.intor("int1e_ovlp")
        Smo = C.T @ S @ C
        np.testing.assert_almost_equal(Smo, 1.0)
        
        # GTO solution and its Laplacian on FEM quadrature
        u_gto, u_Delta_gto = gto.build_Delta(mol, coords, C)

        # GTO solution is normalized, otherwise FEM grid not good
        np.testing.assert_almost_equal(norm.inner(u_gto, u_gto, dV), 1.00)
        
        # Compare to reference kinetic energy from PySCF
        T = mol.intor_symmetric('int1e_kin')
        Ekin_ref = C.T @ T @ C
        Ekin = - 0.5 * norm.inner(u_gto, u_Delta_gto, dV) 
        np.testing.assert_almost_equal(Ekin_ref, Ekin)

        # H(-X) = E(-X) par convention on prend la positive
        if norm.inner(u_fem, u_gto, dV) < 0 :
            u_gto = - u_gto 

        # Potential energy term
        dV_pot = fem.build_dV_pot(helfem_grid, Z1, Z2, wquad)
        Enuc = - Rh**2 * norm.inner(u_gto, u_gto, dV_pot)
        
        # Compare to reference potential energy from PySCF
        V = mol.intor('int1e_nuc')
        Enuc_ref = C.T @ V @ C
        np.testing.assert_almost_equal(Enuc_ref, Enuc)

    def test_lebedev(self):
        """
        Test Lebedev quadrature accuracy for probe function on 1-sphere
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
        eigval_H, orbs_rad, r_rad, w_rad = utils.atomic_energy(atom_file, lmax)
    
        dV_rad = np.diag(np.multiply(np.square(r_rad), w_rad))
        for l in range(lmax+1):
            A = orbs_rad[l] @ dV_rad @ orbs_rad[l].T
            n = A.shape[0]
            np.testing.assert_almost_equal(A, np.eye(n))

        # Create img directory if non-existing
        if (not os.path.exists("img")): os.mkdir("img") 

        # Plot the radial part of eigenvalues
        plt.plot(r_rad, r_rad**2*orbs_rad[0][0]**2, label='1s')
        plt.plot(r_rad, r_rad**2*orbs_rad[0][1]**2, label='2s')
        plt.plot(r_rad, r_rad**2*orbs_rad[0][2]**2, label='3s')
        plt.plot(r_rad, r_rad**2*orbs_rad[0][3]**2, label='4s')
        plt.xlabel("r")
        plt.ylabel(r"$r^2\phi$")
        plt.legend()
        plt.savefig("img/s_hydrogen.pdf")
        plt.close()

        plt.plot(r_rad, r_rad**2*orbs_rad[1][1]**2, label='2px')
        plt.plot(r_rad, r_rad**2*orbs_rad[1][2]**2, label='3px')
        plt.plot(r_rad, r_rad**2*orbs_rad[1][3]**2, label='4px')
        plt.xlabel("r")
        plt.ylabel(r"$r^2\phi$")
        plt.legend()
        plt.savefig("img/px_hydrogen.pdf")
        plt.close()

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

    def greens_function(self):
        """
        Test Laplacian Green's function integration using quadratures
        on Gaussian trial functions whose exact integral is known

        \int_R3 \int_R3 f(x)f(y)/|x-y| dx dy

        for f(x) = c*exp(-|r|^2) where c is normalisation factor
        """

        density_file = 'dat/density_small.hdf5'
        dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = utils.diatomic_density(density_file)
        coords = fem.prolate_to_cart(Rh, helfem_grid)
        ncoords = coords.shape[0]
        
        # Normalised Gaussian function
        f = lambda x: pow(1./np.pi, 3/2) * np.exp(-(x[0]**2 + x[1]**2 + x[2]**2))
        f_vec = lambda x_vec: np.array([f(x) for x in x_vec])
        
        # Coulomb potential centered at zero
        kernel = lambda x: 1./np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

        # Compute <f, Delta^{-1} f>
        integral = norm.green_inner(f_vec, kernel, coords, dV)
        
        # Exact value from Helgaker (bielectronic integral)
        a = 0.5 # a = pq/(p+q), p=q=1
        exact = np.sqrt(4*a/np.pi)
        # integral small:            0.7698504868876277
        # integral small long:       0.7701445190
        # integral small extra long: 0.7690315414 
        # integral medium long:      0.7698993927 
        print("Integral %.10f, exact %.10f" %(integral, exact) )

    def partitions(self):
        """
        Problem avec FEM grid is the quality around zero
        """

        a, b = 0.5, 0.8
        sigmas = (3, 3, 3, 9)

        # HelFEM grid
        density_file = 'dat/density.hdf5'
        dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2 = utils.diatomic_density(density_file)
        coords_init = fem.prolate_to_cart(Rh, helfem_grid)
        print(coords_init.shape)
        coords = np.zeros(coords_init.shape)
        coords[:,2] = np.sort(coords_init[:,2])
        pou.partition_compl(Rh, coords, a, b, plot=True)

        # Uniform grid
        X = np.linspace(-2*Rh, 2*Rh, 500)
        coords = np.zeros((X.shape[0], 3))
        coords[:,2] = X
        pou.partition_compl(Rh, coords, a, b, plot=True)


if __name__ == '__main__':
    unittest.main() 


