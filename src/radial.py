"""
Routines for radial part of the estimator
"""

def gto_residual():

    lapl_ao = dft.numint.eval_ao(mol, coords, deriv=2)

