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

# get error indicators eta_k
val_atom = np.sqrt(2 * estim_atom[idx])
val_lapl = np.sqrt(estim_Delta[idx])

# get error on eigenvalues
estim_eigval = np.array([data[basis[i]]["estim_eigval"] for i in range(n_bas)])
err_eigval = np.array([data[basis[i]]["err_eigval"] for i in range(n_bas)])

"""
Plots estimator vs true error
"""
def main():

    labels = [basis[idx[i]] for i in range(n_bas)]
    plt.xticks(np.arange(n_bas), labels, rotation=45, ha='right', rotation_mode='anchor')
    plt.plot(err_H[idx], '^--', label="approx. error on $\phi_1$")
    plt.plot(err_eigval[idx], 's--', label="approx. error on $\lambda_1$")
    plt.plot(estim[idx], '^-', label=r"estimate on $\phi_1$")
    plt.plot(estim_eigval[idx], 's-', label=r"estimate on $\lambda_1$")
    plt.plot(val_atom, '*-', label=r"$\eta_1 + \eta_2$")
    plt.plot(val_lapl, 'x-', label=r"$\eta_3$")
    plt.yscale("log")
    plt.grid(color='#EEEEEE')
    plt.legend()
    plt.gcf().set_size_inches(4.8, 3)
    plt.savefig("img/norm.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()


