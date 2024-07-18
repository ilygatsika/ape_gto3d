import numpy as np
from pyscf import gto, scf
import matplotlib
from matplotlib import pyplot as plt

bases = ["ccpvdz", "ccpvtz", "ccpvqz", "ccpv5z"]
nbas = len(bases)

vec_R = np.linspace(2, 7, 20)
ndist = np.size(vec_R)
"""
for i in range(nbas):
    for j in range(nbas):

        etot = np.empty(ndist)

        for k in range(ndist):

            bas1, bas2 = bases[i], bases[j]
            R = vec_R[k]

            basis = {'Li': bas1, 'H': bas2}

            mol = gto.M("Li 0 0 0; H %.2f 0 0" %R, spin=1, charge=3, 
                        basis=basis, verbose=0)
            mf = scf.UHF(mol)
            etot[k] = mf.kernel()
            ediss = -3**2/2
        
        plt.axhline(y=ediss, color='k', linestyle='--', linewidth=0.8)
        plt.plot(vec_R, etot, label=bas1[4]+bas2[4])

plt.ylabel('ground state energy')
plt.xlabel("R internuclear distance")
plt.legend()
plt.savefig("img/adaptive_LiH.png", dpi=200)
plt.close()
"""
R = 1.0
Emat = np.empty((nbas,nbas))
for i in range(nbas): # basis for Li
    for j in range(nbas): # basis for H

        bas1, bas2 = bases[i], bases[j]
        basis = {'Li': bas1, 'H': bas2}

        mol = gto.M(f"Li 0 0 {-R}; H 0 0 {R}", spin=1, charge=3, 
                    basis=basis, verbose=0, unit="bohr")
        mf = scf.UHF(mol)
        etot = mf.kernel()
        Emat[i,j] = etot

        print("R=",R, "bas1=",bas1, "bas2=",bas2, "etot=",etot)

plt.imshow(Emat, origin = 'lower', cmap="bwr")
plt.gca().set_xticklabels(bases, rotation=45, ha='right')
plt.gca().set_yticklabels(bases, rotation=45, ha='right')
clb = plt.colorbar(norm='symlog')
plt.xlabel("H basis")
plt.ylabel("Li basis")
plt.xticks(np.arange(nbas))
plt.yticks(np.arange(nbas))
clb.set_label('ground state energy (Hartree)',fontsize=9)
plt.tight_layout()
plt.savefig("img/LiH_map.png", dpi=150)
plt.close()


