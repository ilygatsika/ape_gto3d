Description

    Test of the a posteriori error estimators for Gaussian discretizations of
    the linear Schrodinger-type eigenvalue problem for Coulomb potential.
    Molecular system is diatomic molecule with one electron in three-dimensions.

    Acknowledgement: thanks to Susi Lethola for insightful discussions

Requirements

        >>> python -m pip install -r requirements.txt

Usage

    Main H-norm calculation and estimation:

        >>> python3 main.py

    Input is results obtained with diatomic_1e of HelFEM placed in the dat
    directory. The parameters of diatomic_1e are obtained from diatomic_cbasis.
    See dat/helfem_script.txt.

    Plot results with

        >>> python3 plot.py

Interface with

    PySCF (https://github.com/pyscf/pyscf)
    HelFEM (https://github.com/susilehtola/HelFEM)

TODO
    * run LiH^{3+} in HelFEM for the distance that can be interesting for
      adaptive. Do the study for many basis sets and many distances in order to
      find an interesting distance

