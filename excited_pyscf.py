from pyscf import gto

Rh = 0.4
basis = "cc-pvdz"
mol = gto.M(
        atom=f'H 0 0 {-Rh}; H 0 0 {Rh}', 
        unit='bohr', 
        charge=1, 
        spin=1,
        basis=basis, 
        verbose=0
        )

# Run unrestricted Hartree-Fock
myhf = mol.UHF()
myhf.run()
    
# Get ground-state energy
E_gto_tot = myhf.kernel()
C = myhf.mo_coeff[0][:,0]

lambda_gto = myhf.mo_energy


