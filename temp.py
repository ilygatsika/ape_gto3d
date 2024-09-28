import src.utils as utils
import numpy as np
import pickle
import sys

# User input
basis = str(sys.argv[1]) # GTO basis set

femfile = "dat/helfem.chk"
resfile = "out/res.pickle"   

shift = 4.0 # 3.80688477

# Reference FEM solution from HelFEM
Efem, E2, Efem_kin, Efem_nuc, Efem_nucr = utils.diatomic_energy(femfile)
Efem = Efem - Efem_nucr + shift

# Constant of Assumption 3
cH = 1./np.sqrt(Efem)

print("cH=", cH)

# Store results to file
data = {}
data["cH"] = cH
key = basis
utils.store_to_file(resfile, key, data)


