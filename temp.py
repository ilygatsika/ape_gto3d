import src.fem as fem
import src.read as read
import src.gto as gto
import src.partition as pou
from pyscf.solvent import ddcosmo
from pyscf.symm import sph
from pyscf import dft
import numpy as np
import matplotlib.pyplot as plt

"""
Atoms are at -0.7 and +0.7
Partition parameters vary in -1.4,..,1.4
"""

# Parameters for partition overlap 
amin = 0.5
amax = 1
# Parameters for spectral decomposition
nmax = 4 # lmax <= 15 due to PySCF
shift = 4.0 # 3.80688477

# Input files
density_file = 'dat/density.hdf5'
helfem_res_file = 'dat/helfem.chk'
atom_file = 'dat/1e_lmax20_Rmax1_4.chk'

# Read data Diatomic
dV, Rh, helfem_grid, wquad, u_fem, Z1, Z2, T_fem, P = read.diatomic_density(density_file)
Efem, Efem_kin, Efem_nuc, Efem_nucr, T, S, H0 = read.diatomic_energy(helfem_res_file)

print("u_fem", u_fem[0].shape)
print("T_fem", T_fem[0].shape)
print("T", T.shape)
print("S", S.shape)
print("H0", H0.shape)
print("P", P[0].shape)
print(P[0])
exit()

def inner_projection(u1, u2, dV=dV):
    return np.sum(u1 * u2 * dV)

# Transfrom HelFEM grid to cart
coords = fem.prolate_to_cart(Rh, helfem_grid)

print("u_fem shape", u_fem.shape)
print("coords shape", coords.shape)

# Define PySCF Mole object
basis = 'cc-pvqz'
mol, E_gto, C = gto.build_gto_sol(Rh, basis) 
ao_value = dft.numint.eval_ao(mol, coords)
u_gto = ao_value @ C
# H(-X) = E(-X) par convention on prend la positive
if inner_projection(u_fem, u_gto) < 0 :
    C = - C
    u_gto = - u_gto

# Read data Atomic
# orbs[l][n] is nl (1s,2s,..) size num_1d
E_atom, orbs, vec_r, wr = read.atomic_energy(atom_file, nmax)
num_1d = vec_r.shape[0]
print("Using 1D grid of %i points" % num_1d)
print("Max 1d_grid %.5f" %np.max(vec_r))

print("wquad grid", wquad.shape)
print("radial grid", vec_r.shape)

y = pou.partition_vec(vec_r, amin, amax)
plt.plot(vec_r, y)
plt.savefig("img/partition.pdf")
plt.close()


# Coulomb potential integration
dV_pot = fem.build_dV_pot(helfem_grid, Z1, Z2, wquad)

# Residual evaluated on FEM
Res = gto.residual(mol, coords, C, E_gto, Rh, Z1, Z2, flag)
nfem = np.size(u_fem)
val_pot = - np.power(Rh,2) * np.outer(np.ones(nfem), dV_pot)
print(u_fem.shape)
print(dV_pot.shape)
print((u_fem * dV_pot).shape)
val_pot = - np.power(Rh,2) * np.sum(u_fem * u_fem * dV_pot)
Vnuc = - np.power(Rh,2) * np.diag(dV_pot.flatten())
print(Vnuc.shape)
print(np.sum(Vnuc @ u_fem.T**2))
print(Efem_nuc, val_pot)
S = np.diag(dV.flatten())
print(np.sum(S @ u_fem.T**2))
print("T_fem shape", T_fem.shape)

H0 = np.diag(T_fem) + Vnuc
print(Efem, np.sum(H0 @ u_fem.T**2))
exit()
f = S @ Res
c = np.linalg.solve(H0, f)
print(c.shape)
print("dual norm of residual", c @ f) 

exit()

# Electron-nuclear repulsion
nuc = np.array([0,0, Rh])
Vrad = lambda x: pow(x[0]**2+x[1]**2+x[2]**2,-1)
V = lambda x: Z1*Vrad(x-nuc) + Z2*Vrad(x+nuc)

# Load Lebedev rule on unit sphere
lmax = nmax
lebedev_order = 7
coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(lebedev_order)
ngrid_ang = coords_1sph.shape[0]
print("Lebedev %i" %ngrid_ang)

# unit volume for radial
dV_rad = np.diag(np.multiply(np.square(vec_r), wr))

# unit volume for angular
dV_ang = np.diag(weights_1sph)

# Partition of unity evaluated on radial part
p = pou.partition_vec(vec_r, amin, amax)

# TODO DEBUG

for n in range(1, nmax+1):
    for l in range(n):
        for m in range(-l,l+1):

            Rnl = orbs[l][n]
            # Normalization coefficient
            int_val = Rnl @ dV_rad @ Rnl.T
            norml = 1./int_val
       
            # Angular part evaluated on a grid
            val_r = []
            val_debug = []
            # fix radial part
            for i in range(num_1d):

                # Loop on m=-l,..,l (angular part)
                ylms = sph.real_sph_vec(coords_1sph, l, True)
                ylm_1sph = np.vstack(ylms) # array (m,num_sph)
                num_ang = ylm_1sph.shape
                #print("angular part evaluated on grid of ", num_ang)
             
                # Loop degeneracy
                val = 0
                for k in range(num_ang[0]):

                    psi = ylm_1sph[k] # eigenstate at l,m

                    # Compute angular integral associated to fixed r
                    val += Rnl[i]**2 * (psi @ dV_ang @ psi.T)

                val_r.append(val)

            val_r = np.array(val_r)
            # Compute radial integral
            #val_estim = (norml**2 * val_r @ dV_rad @ val_r.T)**2
            val_debug = (Rnl @ dV_rad @ Rnl.T)
            #print(val_estim)
            print(val_debug)

            for n1 in range(1, nmax+1):
                for l1 in range(n1):
                    for m1 in range(-l1,l1+1):
                        
                        Rnl1 = orbs[l1][n1]
                        val_debug = (Rnl @ dV_rad @ Rnl1.T)
                        print(n,n1,val_debug)




exit()


val_estim = 0

# Eigendecomposition of size nmax
for n in range(1,nmax+1):
 
    # Loop on l = 0,..,n-1
    for l in range(n):

        # Radial on j and l
        """
        orbs[0][0] 1s
        orbs[0][1] 2s
        orbs[0][2] 3s
        """
        Rln = orbs[n][l]
        
        # Normalization coefficient
        int_val = Rln @ dV_rad @ Rln.T
        norml = 1./int_val

        # eigenvalue
        eigval = E_atom[n][0][l] + shift
        print(n, l, "eigvalue", eigval)
       
        # Angular part evaluated on a grid
        val_r = []
        # fix radial part
        for i in range(num_1d):

            # Grid on r-sphere
            r = vec_r[i]

            # Loop on m=-l,..,l (angular part)
            ylms = sph.real_sph_vec(coords_1sph, l, True)
            ylm_1sph = np.vstack(ylms) # array (m,num_sph)
            num_ang = ylm_1sph.shape
            #print("angular part evaluated on grid of ", num_ang)
             
            # Kinetic term evaluated on grid
            u_Delta_gto = gto.build_Delta(mol, coords_1sph, C)

            # Coulomb term
            ao_value = dft.numint.eval_ao(mol, coords_1sph)
            u_gto = ao_value @ C
            V_eval = [V(point) for point in coords_1sph]
            
            # Residual evaluated on given r and grid of theta,phi
            Res = E_gto * u_gto - (- 0.5 * u_Delta_gto - V_eval * u_gto)
            
            # Loop degeneracy
            val = 0
            for k in range(num_ang[0]):

                psi = ylm_1sph[k] # eigenstate at l,m

                # Compute angular integral associated to fixed r
                val += Rln[i] * (Res @ dV_ang @ psi.T)

            val_r.append(val)

        val_r = np.array(val_r)
        # Compute radial integral
        val_estim += 1./eigval * (norml * val_r @ dV_rad @ np.sqrt(p).T)**2

print(val_estim)
exit()


# Plot radial integrand
plt.plot(vec_r, np.square(vec_r) * int_val_L2**2, label=r"pRes*pRes")
plt.plot(vec_r, np.square(vec_r) * int_val**2, label=r"sum_i:pRes*psi_i")
for n in range(1, nmax+1):
    plt.plot(vec_r, np.square(vec_r) * int_val_psi[n]**2, label=r"r**2*psi_%i**2" %n)
#plt.plot(vec_r, np.square(vec_r) * int_val_weight**2, label=r"sum_i:1/ei*pRes*psi_i")
plt.legend()
plt.ylim(0,50)
plt.xlim(0,0.6)
plt.savefig("img/radial_integral.pdf")
plt.close()

# Quadrature on radial part
rad = np.diag(np.multiply(np.square(vec_r), wr))
val = int_val @ rad @ int_val.T
val_weight = int_val_weight @ rad @ int_val_weight.T
val_L2 = int_val_L2 @ rad @ int_val_L2.T
val_debug = int_val_debug @ rad @ int_val_debug.T

c = []
for n in range(1, nmax+1):
    val = int_val_psi[n]
    norm_psi = val @ rad @ val.T
    c.append
    print(n, norm_psi)

print("p_u_gto L2 norm", val_debug)
"""
from scipy import integrate
def f(x,y,z):
    grid = np.array([[x,y,z]])
    ao_value = dft.numint.eval_ao(mol, grid)
    val = ao_value @ C
    r = np.sqrt(x**2 + y**2 + z**2)
    return pou.partition(r, 0.3, 0.5) * val**2
print(integrate.tplquad(f, -1, 1, -1, 1, -1, 1))
"""
eigval_last = -0.5/(nmax+1)**2
val_tot = val_weight + 1./eigval_last * (val_L2 - val_weight)

print(val_tot)






