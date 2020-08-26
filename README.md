TNG_BaryonPasting
==========

This is the repository for the IllustrisTNG data extract code used to provide the required quantities for various Baryon Pasting projects.
Due to the need to for properties that extend well beyond what is traditionally considered a halo (i.e. beyond various spherical overdensity definitions) there is a class for reading all particles in a given snapshot.
This is due to TNG's data store approach, which organises particles into groups defined by a Friend-of-Friend's percolation algorithm and then places particles outside of groups together at the end.
Reading the entire snapshot requires a substantial amount (TBs) of memory due to the numerical resolution and size of the TNG simulations.
Additionally, there is heavy use of MPI functions to split the work and make the computation of many properties for many halos tractable.
Therefore, feel free to use this code, but make sure you have a HPC facility handy.

Requirements
------------

This has been tested and run using `python` `v3.7.3`. It will definitely not run if you are a `python2` user.

### Packages

+ `numpy` - required for the core numerical routines
+ `scipy` - required for minimization routines and KD tree construction
+ `h5py` - required to read data from the IllusttrisTNG HDF5 output files
+ `astropy` - required for computing various cosmological quantities
+ `mpi4py` - used to split the work over available tasks
+ `numba` - used in the computation of the gravitational potential for improved efficiency 
