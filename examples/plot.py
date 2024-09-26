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

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, 
            figsize=(4.5,4.3), gridspec_kw={'height_ratios': [2, 1]})
    labels = [basis[idx[i]] for i in range(n_bas)]
    plt.xticks(np.arange(n_bas), labels, rotation=45, ha='right', rotation_mode='anchor')
    
    ax1.plot(err_H[idx], 's-', label=r"$\|\varphi_1 - \varphi_{1N}\|_A$")
    ax1.plot(estim[idx], '^-', label=r"estimate on $\varphi_1$")
    ax1.plot(val_atom, '*-', label=r"$(\eta^2_1 + \eta^2_2)^{1/2}$")
    ax1.plot(val_lapl, 'x-', label=r"$\eta_3$") 
    ax1.grid(color='#EEEEEE')
    ax1.set_yscale("log")
    ax1.legend()

    ax2.plot(err_eigval[idx], 's--', label="$\lambda_{1N} - \lambda_1$")
    ax2.plot(estim_eigval[idx], '^--', label=r"estimate on $\lambda_1$")
    ax2.grid(color='#EEEEEE')
    ax2.set_yscale("log")
    ax2.legend()
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("img/norm.pdf", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()


