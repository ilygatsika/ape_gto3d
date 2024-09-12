import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

"""
    This module produces figures for estimator of H2+.
"""

# Create img directory if non-existing
if (not os.path.exists("img")): os.mkdir("img") 

"""
Read data
"""
with open("out/res.pickle", 'rb') as file:
	data = pickle.load(file)

basis = list(data.keys())
n_bas = len(basis)
s = data[basis[0]]["shift"]
s1 = data[basis[0]]["shift1"]
s2 = data[basis[0]]["shift2"]
s3 = data[basis[0]]["shift3"]
shift = (s1, s2, s3, s)
print(shift)

estim_atom = np.array([data[basis[i]]["estim_atom"] for i in range(n_bas)])
estim_Delta = np.array([data[basis[i]]["estim_Delta"] for i in range(n_bas)])
estim = np.array([data[basis[i]]["estimator"] for i in range(n_bas)])
err_H = np.array([data[basis[i]]["err_H"] for i in range(n_bas)])

# Sort with desceanding error
idx = np.argsort(err_H)[::-1]

"""
Plots estimator vs true error
"""
def main():

    labels = [basis[idx[i]] for i in range(n_bas)]
    plt.xticks(np.arange(n_bas), labels, rotation=45, fontsize=12, ha='right', rotation_mode='anchor')
    plt.plot(estim[idx], 'x-', label=r"estimator")
    plt.plot(estim_atom[idx], 'x-', label=r"atomic")
    plt.plot(estim_Delta[idx], 'x-', label=r"Laplacian")
    plt.plot(err_H[idx], '^-', label="approx. error")
    plt.yscale("log")
    plt.legend()
    plt.gcf().set_size_inches(7, 6)
    plt.savefig("img/norm.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()


