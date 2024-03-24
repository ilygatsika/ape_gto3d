Description

    Test of the a posteriori error estimators for Gaussian discretizations of
    the linear Schrodinger-type eigenvalue problem for Coulomb potential.
    Molecular system is diatomic molecule with one electron in three-dimensions.

    Thanks to Susi Lethola for useful indications on using HelFEM within the
    present code.

Usage

    Main H-norm calculation and estimation:

        >>> python3 main.py

    Input is results obtained with diatomic_1e of HelFEM placed in the dat
    directory. The parameters of diatomic_1e are obtained from diatomic_cbasis.
    See dat/helfem_script.txt.

Requires

    PySCF (https://github.com/pyscf/pyscf)
    HelFEM (https://github.com/susilehtola/HelFEM)

TODO 

    [OK] ce code va être dans un répertoire git
    [*] solve bug on H-norm error formula
    [*] obtain atomic operator eigenvectors (understand R_nl)
    [*] use the spherical harmonics from PySCF
    [*] obtain radial part of eigenfunctions evaluated on FEM grid 
    [*] test precision of lebedev quadrature on sphere


