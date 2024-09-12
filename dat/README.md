# Data sets

This directory gathers data sets computed with the HelFEM (https://github.com/susilehtola/HelFEM.git) program. The data contains finite element solutions to molecular and atomic eigenproblems for the Coulomb potential.

Contents of checkpoint files can be printed in a terminal with
```
h5dump --header dat/<filename>.chk
```
The following simulations generate the data.

## H2+

### Fine grid

An estimated accuracy of 1e-10 is achieved with this quadrature grid on the molecule

```
./diatomic_cbasis --Z1=H --Z2=H --Rbond=1.4
./diatomic_1e --Z1=H --Z2=H --Rbond=1.4 --angstrom=0 --grid=4 --zexp=1.0 --primbas=4 --nnodes=15 --nelem=5 --Rmax=40 --lmax=6 -save=helfem.chk
./diatomic_dgrid --load=helfem_LiH.chk --save=density_LiH.hdf5
./1e_atom --Z=H --nelem=5 --lmax=20 --save=1e_lmax20_Rmax1_4.chk
```

This generates `helfem.chk`, `density.hdf5` and `1e_lmax20_Rmax1_4.chk`.

### Coarse grid

Fewer grid points useful for running small tests

```
./diatomic_1e --Z1=H --Z2=H --Rbond=1.4 --angstrom=0 --grid=4 --zexp=1.0--primbas=3 --nnodes=2 --nelem=3 --Rmax=40 --lmax=3 --save=helfem_small.chk
./diatomic_dgrid --load=helfem_small.chk --save=density_small.hdf5
```

This generates `helfem_small.chk` and `density_small.hdf5`.

## LiH^{3+} 

An estimated accuracy of 1e-10 is achieved with this quadrature grid on the molecule

```
./diatomic_cbasis --Z1=Li --Z2=H --Rbond=2.0
./diatomic_1e --Z1=Li --Z2=H --Rbond=2.0 --angstrom=0 --grid=4 --zexp=1.0 --primbas=4 --nnodes=15 --nelem=5 --Rmax=4.0 --lmax=10 --save=helfem_LiH.chk
./diatomic_dgrid --load=helfem_LiH.chk --save=density_LiH.hdf5
./1e_atom --Z=Li --nelem=5 --lmax=10 --save=1e_Li_in_LiH.chk
./1e_atom --Z=H --nelem=5 --lmax=10 --save=1e_H_in_LiH.chk
```

This generates `helfem_LiH.chk`, `density_LiH.hdf5`, `1e_Li_in_LiH.chk` and `1e_H_in_LiH.chk`.


