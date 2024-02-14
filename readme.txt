Description

    Test of the a posteriori error estimators for Gaussian discretizations of
    the linear Schrodinger-type eigenvalue problem for Coulomb potential.
    Molecular system is diatomic molecule with one electron in three-dimensions.

Usage

    Main H-norm calculation and estimation:

        >>> python3 main.py

    Input is results obtained with diatomic_1e of HelFEM placed in the dat
    directory. The parameters of diatomic_1e are obtained from diatomic_cbasis.
    See dat/helfem_script.txt.

Requires

    PySCF, HelFEM

TODO 

    [OK] ce code va être dans un répertoire git
    [*] solve bug on H-norm error formula
    [*] use the spherical harmonics from PySCF
    [*] write a bash script that calls the diatomic_cbasis then the diatomic
    code with the appropriate parameters and then outputs the density file


