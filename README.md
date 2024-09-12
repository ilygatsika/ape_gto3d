# ape_gto3d

Python code for the a posteriori estimation of Gaussian-type orbital basis set errors on diatomic molecules with one-electron. This code is used in our paper _A posteriori error estimates for Schrödinger operators discretized with linear combinations of atomic orbital_ available on [arXiv](pending).

## Requirements

Python xxx with the libraries:
- xx for testing;
- PySCF (https://github.com/pyscf/pyscf) for GTO computations;
- xx for saving results;
- xx for plotting results.

## Install and test

In the root directory, run
```
python -m pip install -r requirements.txt
python test.py
```

## Usage

```
./run.sh
```

This program runs all calculations at once, stores results in json format and then generates figures in the `img` directory. Note that the `out` directory of the present git repo already contains all precalculated results. Attention : running all the computations from scratch takes around 1 week on a supercomputer and allocates more than 20 gigs of memory. 

Input data is finite element solutions used as a reference and obtained using HelFEM (https://github.com/susilehtola/HelFEM).

## Authors

Mi-Song Dupuy (Sorbonne Université), Geneviève Dusson (Université de Franche Comté, CNRS), Ioanna-Maria Lygatsika (Sorbonne Université).

## Credits

Contributor: Susi Lehtola (University of Helsinki)

Support of the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement EMC2 No 810367).

TODO 
- use commands to generate HelFEM data and diff that it is the same as precomputed ones


