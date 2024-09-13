# ape_gto3d

Python code for the a posteriori estimation of Gaussian-type orbital basis set errors on diatomic molecules with one-electron. This code is used in our paper _A posteriori error estimates for Schrödinger operators discretized with linear combinations of atomic orbitals_ available on [arXiv](pending).

## Requirements

Python 3.10.12 with the libraries:
- numpy, scipy, sympy, PySCF (https://github.com/pyscf/pyscf) for computations;
- h5py for saving results;
- matplotlib for plotting results.

## Install and test

In the root directory, run
```
make install
make test
```

## Usage

```
python3 run.py
```

This program runs all calculations at once, stores results in pickle format and then generates figures in the `img` directory. Note that the `out` directory of the present git repo already contains all precalculated results that enable to reproduce figures. The input data in the `dat` directory contains finite element solutions used as a reference and obtained by HelFEM (https://github.com/susilehtola/HelFEM). 

_Disclaimer:_ Running all the computations from scratch takes around 1 week on a supercomputer and allocates more than 20 gigs of memory. Use with caution. Dedicated script for long computations:

```
./run_long.sh
```

## Authors

Mi-Song Dupuy (Sorbonne Université), Geneviève Dusson (Université de Franche Comté, CNRS), Ioanna-Maria Lygatsika (Sorbonne Université).

## Credits

Contributor: Susi Lehtola (University of Helsinki)

Support of the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement EMC2 No 810367).


