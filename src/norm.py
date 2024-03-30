import numpy as np
from pyscf.solvent import ddcosmo
from pyscf.symm import sph
import src.gto

"""
Routines for evaluating norms and inner products
of functions from R3 to R
"""

def inner(u1, u2, dV):
    """
    L2 inner product with dV quadrature weights
    """
    return np.sum(u1 * u2 * dV)

def green_inner(f, kernel, coords, dV):
    """
    Return scalar equal to <f, G(x,y) f>_{L_2}
    using Green kernel discretized on coords
    """

    # Evaluate 3D integral on y
    ncoords = coords.shape[0]
    f_y_1 = np.empty(ncoords, dtype=float)
    f_z_2 = np.array([kernel(z) for z in coords])
    for i in range(ncoords):

        # Evaluate 3D integral on z
        f_y_1[i] = inner(f(coords + coords[i]), f_z_2, dV)

    f_y_2 = f(coords)
    integral = inner(f_y_1, f_y_2, dV)

    return integral 

def atom_inner(f, g, eigpairs, rad_grid, lebedev_order, lmax, shift):
    """
    Spectral decomposition of order lmax for given eigenpairs
    for dual norm of f evaluated using Lebedev quadratures
    """

    E, orbs = eigpairs
    r_rad, w_rad = rad_grid
    
    # Load Lebedev rule on unit sphere
    r_1sph, w_1sph = ddcosmo.make_grids_one_sphere(lebedev_order)
    npt_1sph = r_1sph.shape[0]

    # Integration volume for radial and angular
    num_1d = r_rad.shape[0]
    dV_rad = np.diag(np.multiply(np.square(r_rad), w_rad))
    dV_ang = np.diag(w_1sph)

    # Precompute spherical harmonics
    ylms = sph.real_sph_vec(r_1sph, lmax+1, False)

    # Precompute function on grid of r-spheres
    f_rsph = np.empty((num_1d, npt_1sph), dtype=float)
    for i in range(num_1d):
        r_rsph = r_rad[i] * r_1sph
        f_rsph[i] = f(r_rsph)

    estim_atom = 0 # spectral decomposition term
    estim_atom_1 = 0 # remaining term, without 1/eigval
    eigval_max = -1

    # Eigendecomposition of size lmax
    for n in range(lmax+1):
 
        # Loop on l = 0,..,n-1
        for l in range(n):

            # Recover spherical harmonic
            Rln = orbs[l][n]
            yl_vec_m = ylms[l]
            m_comp = yl_vec_m.shape[0]
        
            # Eigenvalue
            eigval = E[l][0][n] + shift
            # store largest eigenvalue
            if (eigval > eigval_max): 
                eigval_max = eigval

            # Angular part evaluated on a grid
            int_rad = np.empty(num_1d, dtype=float)
            # fix radial part
            for i in range(num_1d):

                # Grid on r-sphere
                radius = r_rad[i]
                r_rsph = radius * r_1sph
            
                # Function on the r-sphere
                fval = f_rsph[i]

                # Loop degeneracy on m
                val_m = np.array([fval @ dV_ang @ yl_vec_m[m].T for m in range(m_comp)])
                int_rad[i] = Rln[i] * np.sum(val_m)

            # Compute radial integral
            integral = (int_rad @ dV_rad @ g.T)**2
            estim_atom += 1./eigval * integral
            estim_atom_1 += integral
   
    # L2 norm
    int_rad = np.empty(num_1d)
    for i in range(num_1d):
        fval = f_rsph[i]
        int_rad[i] = fval @ dV_ang @ fval.T

    inner_L2 = int_rad @ dV_rad @ (g**2).T
    
    # Form atomic part of Theorem 3.7
    tot_estim_atom = estim_atom + 1./eigval_max * (inner_L2 - estim_atom_1)
    
    return tot_estim_atom

def H_norm(mol, coords, C, u_fem, shift):
    """
    Evaluate |ufem - ugto|_H in energy norm
    """

    u_Delta_gto = gto.build_Delta(mol, coords, C)
    val_Delta = inner(u_fem, u_Delta_gto)

    err_H = np.sqrt(Efem + E_gto - 2*( - 0.5 * val_Delta - val_pot + shift))

    return err_H

